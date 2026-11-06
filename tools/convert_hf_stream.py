#!/usr/bin/env python3
"""Low-RAM streaming HF safetensors → rabbit-ref-v2 (group-int4 or f32 experts).

Supports single-file and multi-shard safetensors without loading the full
checkpoint into RAM.

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

try:
    from safetensors import safe_open
except ImportError:
    print("pip install safetensors numpy", file=sys.stderr)
    raise SystemExit(1)


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


class ShardIndex:
    """Map tensor name → shard path; load tensors on demand."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.name_to_file: Dict[str, Path] = {}
        self._open: Dict[str, object] = {}

        st_files = sorted(model_dir.glob("*.safetensors"))
        if not st_files:
            raise SystemExit(f"No safetensors in {model_dir}")

        index_path = model_dir / "model.safetensors.index.json"
        if index_path.is_file():
            idx = json.loads(index_path.read_text())
            weight_map = idx.get("weight_map") or {}
            for name, fname in weight_map.items():
                self.name_to_file[name] = model_dir / fname
            print(
                f"[index] {len(self.name_to_file)} tensors across "
                f"{len(set(self.name_to_file.values()))} shards"
            )
        else:
            for path in st_files:
                with safe_open(str(path), framework="np") as st:
                    for key in st.keys():
                        self.name_to_file[key] = path
            print(
                f"[index] {len(self.name_to_file)} tensors in "
                f"{len(st_files)} file(s)"
            )

    def has(self, name: str) -> bool:
        return name in self.name_to_file

    def get(self, name: str) -> Optional[np.ndarray]:
        path = self.name_to_file.get(name)
        if path is None:
            return None
        key = str(path)
        if key not in self._open:
            self._open[key] = safe_open(str(path), framework="np")
        st = self._open[key]
        return np.asarray(st.get_tensor(name), dtype=np.float32)

    def require(self, name: str) -> np.ndarray:
        t = self.get(name)
        if t is None:
            sample = list(self.name_to_file)[:8]
            raise SystemExit(
                f"Missing required tensor: {name}\n"
                f"  Available sample: {sample}..."
            )
        return t

    def require_any(self, names) -> Tuple[str, np.ndarray]:
        name, val = first_present(self.get, names)
        if val is None:
            sample = list(self.name_to_file)[:8]
            raise SystemExit(
                f"Missing required tensor (tried {list(names)[:4]}…)\n"
                f"  Available sample: {sample}..."
            )
        return name, val  # type: ignore[return-value]

    def close(self) -> None:
        self._open.clear()


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
    """Load gate/up/down for one routed expert; return arrays + matching scheme index."""
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
    """Load DeepSeek-style always-on shared_experts MLP for one layer.

    HF may fuse width as moe_intermediate * n_shared; we crop/pad to inter×H.
    """
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
    ap.add_argument(
        "--name",
        default=None,
        help="Model name stored in header (default: model-dir basename)",
    )
    ap.add_argument(
        "--quant",
        type=int,
        default=4,
        choices=[0, 4],
        help="Expert quant: 0=f32, 4=group-int4 (default)",
    )
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
    # Prefer tensors when present; fall back to config count (at least 1 if tensors exist)
    n_shared = 0
    if has_shared:
        n_shared = max(1, n_shared_cfg)
    elif n_shared_cfg > 0:
        # Config claims shared experts but tensors not found under known names
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
                # HF typically stores one fused shared MLP; replicate S times if S>1
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
