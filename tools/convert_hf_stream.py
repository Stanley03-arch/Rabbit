#!/usr/bin/env python3
"""Low-RAM streaming HF safetensors → rabbit-ref-v2 (group-int4 or f32 experts).

Supports single-file and multi-shard safetensors without loading the full
checkpoint into RAM. Uses a pure streaming reader (header parse + seek/read);
peak process RAM is approximately the size of the largest single tensor,
not the whole file. No full-file mmap and no dependency on the safetensors package.

Expert / router naming (auto-detected):
  Mixtral:  block_sparse_moe.experts.{e}.w1|w2|w3  + block_sparse_moe.gate
  Qwen:     mlp.experts.{e}.gate_proj|up_proj|down_proj  + mlp.gate
  DeepSeek: mlp.experts.* + mlp.shared_experts.gate/up/down_proj
            (config: n_routed_experts, n_shared_experts)
  Other:    moe.experts / sparse_moe variants (see rabbit_name_maps.py)

Disk layout when n_shared > 0:
  per layer: [E routed][S shared]  →  L * (E + S) expert records
  header.n_experts = E (router width)
  reserved[12:16]  = S
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rabbit_convert_pack import write_header
from rabbit_name_maps import (
    ATTN_K_TEMPLATES,
    ATTN_O_TEMPLATES,
    ATTN_Q_TEMPLATES,
    ATTN_V_TEMPLATES,
    EMBED_TEMPLATES,
    FINAL_NORM_TEMPLATES,
    INPUT_NORM_TEMPLATES,
    LM_HEAD_TEMPLATES,
    POST_NORM_TEMPLATES,
    detect_expert_scheme,
    detect_shared_experts,
    first_present,
    resolve_expert_triple_names,
    resolve_router_name,
    resolve_shared_expert_names,
)

# Pure streaming reader — no full-file mmap. Peak RAM ≈ largest single tensor.


def pack_int4_grouped_fast(mat: np.ndarray, group_size: int = 32) -> bytes:
    flat = np.asarray(mat, dtype=np.float32).ravel()
    n = flat.size
    n_groups = (n + group_size - 1) // group_size
    pad = n_groups * group_size - n
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    groups = flat.reshape(n_groups, group_size)
    max_abs = np.max(np.abs(groups), axis=1)
    scales = (max_abs / 7.0 + 1e-8).astype(np.float32)
    q = np.clip(np.rint(groups / scales[:, None]), -8, 7).astype(np.int16)
    q = q.ravel()[:n]
    if n % 2:
        q = np.concatenate([q, np.zeros(1, dtype=np.int16)])
    lo = (q[0::2] + 8).astype(np.uint8)
    hi = (q[1::2] + 8).astype(np.uint8)
    packed = (lo | (hi << 4)).astype(np.uint8)
    return struct.pack("<I", group_size) + scales.tobytes() + packed.tobytes()


def fit2d(m, rows, cols):
    out = np.zeros((rows, cols), dtype=np.float32)
    if m is None:
        return out
    m = np.asarray(m, dtype=np.float32)
    if m.ndim != 2:
        return out
    r, c = min(rows, m.shape[0]), min(cols, m.shape[1])
    out[:r, :c] = m[:r, :c]
    return out


def expert_nbytes(inter, H, quant: int = 4, gs=32):
    if quant == 0:
        return (inter * H + inter * H + H * inter) * 4

    def one(n):
        ng = (n + gs - 1) // gs
        return 4 + ng * 4 + (n + 1) // 2

    return one(inter * H) * 2 + one(H * inter)


_DTYPE_MAP = {
    "F32": (np.float32, 4),
    "F16": (np.float16, 2),
    "BF16": (None, 2),
    "I8": (np.int8, 1),
    "U8": (np.uint8, 1),
    "I16": (np.int16, 2),
    "I32": (np.int32, 4),
    "I64": (np.int64, 8),
    "F64": (np.float64, 8),
    "BOOL": (np.bool_, 1),
}


def _bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def _parse_safetensors_header(path: Path) -> Tuple[Dict[str, dict], int]:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        if header_len > 100 * 1024 * 1024:
            raise SystemExit(f"Implausible safetensors header size {header_len} in {path}")
        header_bytes = f.read(header_len)
    meta = json.loads(header_bytes.decode("utf-8"))
    data_offset = 8 + header_len
    tensors = {k: v for k, v in meta.items() if k != "__metadata__"}
    return tensors, data_offset


class ShardIndex:
    """Map tensor name → metadata; load via seek+read (no full-file mmap)."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._index: Dict[str, Tuple[Path, str, Tuple[int, ...], int, int]] = {}
        self.name_to_file: Dict[str, Path] = {}

        st_files = sorted(model_dir.glob("*.safetensors"))
        if not st_files:
            raise SystemExit(f"No safetensors in {model_dir}")

        for path in st_files:
            sz = path.stat().st_size
            if sz < 64:
                raise SystemExit(
                    f"Safetensors file is empty or tiny ({sz} bytes): {path}\n"
                    "  Hugging Face Xet/CDN downloads often leave 0-byte placeholders.\n"
                    "  Retry with: curl -L -o <file> "
                    "https://huggingface.co/.../resolve/main/model.safetensors\n"
                    "  or set HF_TOKEN and re-run the download."
                )
            if sz > 500 * 1024 * 1024:
                print(
                    f"[info] large shard {path.name} ({sz / 1e9:.2f} GB) — "
                    "using streaming seek/read (no full mmap)."
                )

        index_path = model_dir / "model.safetensors.index.json"
        weight_map: Dict[str, str] = {}
        if index_path.is_file():
            idx = json.loads(index_path.read_text())
            weight_map = idx.get("weight_map") or {}

        files_needed = (
            sorted({model_dir / fname for fname in weight_map.values()})
            if weight_map
            else st_files
        )

        for path in files_needed:
            if not path.is_file():
                raise SystemExit(f"Shard missing: {path}")
            tensors, data_offset = _parse_safetensors_header(path)
            for name, info in tensors.items():
                if weight_map and name not in weight_map:
                    continue
                dtype = info["dtype"]
                shape = tuple(info["shape"])
                start, end = info["data_offsets"]
                nbytes = end - start
                abs_off = data_offset + start
                self._index[name] = (path, dtype, shape, abs_off, nbytes)
                self.name_to_file[name] = path

        print(
            f"[index] {len(self._index)} tensors across "
            f"{len(set(p for p, *_ in self._index.values()))} file(s)  "
            f"(streaming reader)"
        )

    def has(self, name: str) -> bool:
        return name in self._index

    def get(self, name: str) -> Optional[np.ndarray]:
        entry = self._index.get(name)
        if entry is None:
            return None
        path, dtype_str, shape, offset, nbytes = entry
        with open(path, "rb") as f:
            f.seek(offset)
            raw = f.read(nbytes)
        if len(raw) != nbytes:
            raise SystemExit(
                f"Short read for {name}: got {len(raw)} expected {nbytes} from {path}"
            )
        if dtype_str == "BF16":
            arr = _bf16_to_f32(raw).reshape(shape)
        else:
            np_dtype, _itemsize = _DTYPE_MAP.get(dtype_str, (None, 0))
            if np_dtype is None:
                raise SystemExit(f"Unsupported dtype {dtype_str} for tensor {name}")
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
        return arr

    def require(self, name: str) -> np.ndarray:
        t = self.get(name)
        if t is None:
            sample = list(self._index)[:8]
            raise SystemExit(
                f"Missing required tensor: {name}\n  Available sample: {sample}..."
            )
        return t

    def require_any(self, names) -> Tuple[str, np.ndarray]:
        name, val = first_present(self.get, names)
        if val is None:
            sample = list(self._index)[:8]
            raise SystemExit(
                f"Missing required tensor (tried {list(names)[:4]}…)\n"
                f"  Available sample: {sample}..."
            )
        return name, val  # type: ignore[return-value]

    def close(self) -> None:
        pass


def pack_expert(g: np.ndarray, u: np.ndarray, d: np.ndarray, quant: int) -> bytes:
    if quant == 0:
        return (
            g.astype(np.float32).tobytes()
            + u.astype(np.float32).tobytes()
            + d.astype(np.float32).tobytes()
        )
    return (
        pack_int4_grouped_fast(g)
        + pack_int4_grouped_fast(u)
        + pack_int4_grouped_fast(d)
    )


def write_padded_expert(f, blob: bytes, expert_size: int, where: str) -> None:
    if len(blob) < expert_size:
        blob = blob + b"\0" * (expert_size - len(blob))
    elif len(blob) > expert_size:
        raise SystemExit(f"expert blob too large: {len(blob)} > {expert_size} ({where})")
    f.write(blob)


def load_expert(
    shards: ShardIndex,
    layer: int,
    expert: int,
    inter: int,
    H: int,
    scheme_index: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    triples = resolve_expert_triple_names(layer, expert)
    order = list(range(len(triples)))
    if scheme_index is not None:
        order = [scheme_index] + [i for i in order if i != scheme_index]

    for idx in order:
        g_name, u_name, d_name = triples[idx]
        g = shards.get(g_name)
        u = shards.get(u_name)
        d = shards.get(d_name)
        if g is not None and u is not None and d is not None:
            return fit2d(g, inter, H), fit2d(u, inter, H), fit2d(d, H, inter), idx

    tried = [t[0] for t in triples[:4]]
    raise SystemExit(
        f"Missing expert weights for layer={layer} expert={expert}\n"
        f"  Tried schemes starting with: {tried}"
    )


def load_shared_expert(
    shards: ShardIndex,
    layer: int,
    inter: int,
    H: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    for g_name, u_name, d_name in resolve_shared_expert_names(layer):
        g = shards.get(g_name)
        u = shards.get(u_name)
        d = shards.get(d_name)
        if g is not None and u is not None and d is not None:
            return fit2d(g, inter, H), fit2d(u, inter, H), fit2d(d, H, inter)
    raise SystemExit(
        f"Missing shared_experts weights for layer={layer}\n"
        f"  Tried: {[t[0] for t in resolve_shared_expert_names(layer)[:3]]}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stream HF MoE (Mixtral / Qwen / DeepSeek) → .rabbit (low RAM)"
    )
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--quant", type=int, default=4, choices=[0, 4])
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"No config.json in {model_dir}")

    config = json.loads(config_path.read_text())
    H = int(config["hidden_size"])
    L = int(config["num_hidden_layers"])
    V = int(config["vocab_size"])
    E = int(
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("n_routed_experts")
        or 0
    )
    if E <= 0:
        raise SystemExit(
            "config: num_local_experts / num_experts / n_routed_experts missing"
        )
    n_shared_cfg = int(config.get("n_shared_experts") or 0)
    top_k = int(config.get("num_experts_per_tok") or config.get("moe_top_k") or 2)
    inter = int(
        config.get("moe_intermediate_size")
        or config.get("intermediate_size")
        or H * 2
    )
    n_heads = int(config["num_attention_heads"])
    n_kv = int(config.get("num_key_value_heads") or n_heads)
    head_dim = int(config.get("head_dim") or H // n_heads)
    q_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    quant = int(args.quant)
    model_name = (args.name or model_dir.name)[:60]

    shards = ShardIndex(model_dir)
    scheme = detect_expert_scheme(shards.has, layer=0, expert=0)
    has_shared = detect_shared_experts(shards.has, layer=0)
    n_shared = 0
    if has_shared:
        n_shared = max(1, n_shared_cfg)
    elif n_shared_cfg > 0:
        print(
            f"[warn] config n_shared_experts={n_shared_cfg} but no shared_experts tensors found"
        )

    expert_size = expert_nbytes(inter, H, quant=quant)
    dense_size = (V * H + H * V + H) * 4
    dense_size += L * (
        H * 2 + q_dim * H + kv_dim * H + kv_dim * H + H * q_dim + E * H
    ) * 4
    n_records = L * (E + n_shared)

    print(
        f"[config] name={model_name!r} H={H} L={L} V={V} E={E} shared={n_shared} "
        f"inter={inter} heads={n_heads}/{n_kv}/{head_dim} quant={quant}"
    )
    print(
        f"[plan] dense={dense_size/1e6:.1f}MB expert={expert_size/1e3:.1f}KB "
        f"records={n_records} (L*(E+S))"
    )
    print(f"[scheme] experts={scheme or 'auto-probe per layer'}  shared={has_shared}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    scheme_index: Optional[int] = None

    with open(out, "wb") as f:
        write_header(
            f,
            n_layers=L,
            hidden_dim=H,
            vocab_size=V,
            n_experts=E,
            experts_per_token=top_k,
            expert_dim=inter,
            dense_size=dense_size,
            expert_size=expert_size,
            dense_quant=0,
            expert_quant=quant,
            name=model_name,
            n_expert_records=n_records,
            n_heads=n_heads,
            n_kv_heads=n_kv,
            head_dim=head_dim,
            n_shared_experts=n_shared,
        )

        _, embed = shards.require_any(list(EMBED_TEMPLATES))
        emb = fit2d(embed, V, H)
        f.write(emb.tobytes())

        _, lm = first_present(shards.get, LM_HEAD_TEMPLATES)
        if lm is not None:
            lm_w = fit2d(lm, V, H).T.copy()
            f.write(lm_w.astype(np.float32).tobytes())
            del lm, lm_w
        else:
            f.write(emb.T.copy().tobytes())
        del embed, emb

        _, norm = shards.require_any(FINAL_NORM_TEMPLATES)
        norm = np.asarray(norm, dtype=np.float32).reshape(-1)
        if norm.size < H:
            norm = np.pad(norm, (0, H - norm.size), constant_values=1.0)
        f.write(norm[:H].tobytes())

        for layer in range(L):

            def layer_names(templates):
                return [t.format(layer=layer) for t in templates]

            _, in_n = shards.require_any(layer_names(INPUT_NORM_TEMPLATES))
            in_n = np.asarray(in_n, dtype=np.float32).reshape(-1)
            if in_n.size < H:
                in_n = np.pad(in_n, (0, H - in_n.size), constant_values=1.0)
            f.write(in_n[:H].tobytes())

            _, post_n = shards.require_any(layer_names(POST_NORM_TEMPLATES))
            post_n = np.asarray(post_n, dtype=np.float32).reshape(-1)
            if post_n.size < H:
                post_n = np.pad(post_n, (0, H - post_n.size), constant_values=1.0)
            f.write(post_n[:H].tobytes())

            _, q = shards.require_any(layer_names(ATTN_Q_TEMPLATES))
            _, k = shards.require_any(layer_names(ATTN_K_TEMPLATES))
            _, v = shards.require_any(layer_names(ATTN_V_TEMPLATES))
            _, o = shards.require_any(layer_names(ATTN_O_TEMPLATES))
            f.write(fit2d(q, q_dim, H).tobytes())
            f.write(fit2d(k, kv_dim, H).tobytes())
            f.write(fit2d(v, kv_dim, H).tobytes())
            f.write(fit2d(o, H, q_dim).tobytes())

            r_name, router = first_present(shards.get, resolve_router_name(layer))
            if router is None:
                raise SystemExit(
                    f"Missing router/gate for layer {layer} "
                    f"(tried Mixtral / Qwen / moe gate names)"
                )
            f.write(fit2d(router, E, H).tobytes())
            print(f"  dense {layer + 1}/{L}  router={r_name}", flush=True)

        for layer in range(L):
            for e in range(E):
                g, u, d, scheme_index = load_expert(
                    shards, layer, e, inter, H, scheme_index
                )
                blob = pack_expert(g, u, d, quant)
                write_padded_expert(f, blob, expert_size, f"L={layer} e={e}")
                del g, u, d, blob

            for s in range(n_shared):
                g, u, d = load_shared_expert(shards, layer, inter, H)
                blob = pack_expert(g, u, d, quant)
                write_padded_expert(f, blob, expert_size, f"L={layer} shared={s}")
                del g, u, d, blob

            elapsed = time.time() - t0
            extra = f" +{n_shared} shared" if n_shared else ""
            print(f"  experts {layer + 1}/{L}{extra}  ({elapsed:.1f}s)", flush=True)

    shards.close()
    elapsed = time.time() - t0
    print(f"[done] {out}  {out.stat().st_size / 1e6:.1f} MB  in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
