#!/usr/bin/env python3
"""Low-RAM streaming HF safetensors → rabbit-ref-v2 (group-int4 or f32 experts).

Supports single-file and multi-shard safetensors without loading the full
checkpoint into RAM. Uses Mixtral tensor names (w1/w3/w2, block_sparse_moe).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rabbit_convert_pack import write_header

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
        # gate[I,H] + up[I,H] + down[H,I] as f32
        return (inter * H + inter * H + H * inter) * 4

    def one(n):
        ng = (n + gs - 1) // gs
        return 4 + ng * 4 + (n + 1) // 2

    return one(inter * H) * 2 + one(H * inter)


class ShardIndex:
    """Map tensor name → (shard_path, key) without loading weights."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.name_to_file: Dict[str, Path] = {}
        self._open: Dict[str, object] = {}

        st_files = sorted(model_dir.glob("*.safetensors"))
        if not st_files:
            raise SystemExit(f"No safetensors in {model_dir}")

        # Prefer index json if present (HF multi-shard)
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.is_file():
            idx = json.loads(index_path.read_text())
            weight_map = idx.get("weight_map") or {}
            for name, fname in weight_map.items():
                self.name_to_file[name] = model_dir / fname
            print(f"[index] {len(self.name_to_file)} tensors across "
                  f"{len(set(self.name_to_file.values()))} shards")
        else:
            for path in st_files:
                with safe_open(str(path), framework="np") as st:
                    for key in st.keys():
                        self.name_to_file[key] = path
            print(f"[index] {len(self.name_to_file)} tensors in "
                  f"{len(st_files)} file(s)")

    def get(self, name: str) -> np.ndarray:
        path = self.name_to_file.get(name)
        if path is None:
            # try common alternates
            for alt in (
                name.replace("model.", ""),
                "model." + name if not name.startswith("model.") else name,
            ):
                path = self.name_to_file.get(alt)
                if path is not None:
                    name = alt
                    break
        if path is None:
            raise KeyError(name)

        key = str(path)
        if key not in self._open:
            self._open[key] = safe_open(str(path), framework="np")
        st = self._open[key]
        return np.asarray(st.get_tensor(name), dtype=np.float32)

    def require(self, name: str) -> np.ndarray:
        try:
            return self.get(name)
        except KeyError:
            raise SystemExit(
                f"Missing required tensor: {name}\n"
                f"  Available sample: {list(self.name_to_file)[:8]}..."
            )

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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stream HF Mixtral-style MoE → .rabbit (low RAM)"
    )
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--quant", type=int, default=4, choices=[0, 4],
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
    E = int(config.get("num_local_experts") or config.get("num_experts") or 0)
    if E <= 0:
        raise SystemExit("config: num_local_experts / num_experts missing")
    top_k = int(config.get("num_experts_per_tok") or 2)
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

    expert_size = expert_nbytes(inter, H, quant=quant)
    dense_size = (V * H + H * V + H) * 4
    dense_size += L * (
        H * 2 + q_dim * H + kv_dim * H + kv_dim * H + H * q_dim + E * H
    ) * 4
    print(
        f"[config] H={H} L={L} V={V} E={E} inter={inter} "
        f"heads={n_heads}/{n_kv}/{head_dim} quant={quant}"
    )
    print(
        f"[plan] dense={dense_size/1e6:.1f}MB expert={expert_size/1e3:.1f}KB "
        f"records={L*E}"
    )

    shards = ShardIndex(model_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    T = shards.require

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
            name=model_dir.name[:60],
            n_expert_records=L * E,
            n_heads=n_heads,
            n_kv_heads=n_kv,
            head_dim=head_dim,
        )

        embed = T("model.embed_tokens.weight")
        emb = fit2d(embed, V, H)
        f.write(emb.tobytes())
        # Prefer dedicated lm_head if present; else tied embed
        try:
            lm = T("lm_head.weight")
            lm_w = fit2d(lm, V, H).T  # want [H, V]
            if lm_w.shape != (H, V):
                lm_w = emb.T.copy()
            f.write(lm_w.astype(np.float32).tobytes())
            del lm, lm_w
        except SystemExit:
            f.write(emb.T.copy().tobytes())
        del embed, emb

        f.write(T("model.norm.weight").astype(np.float32).reshape(-1)[:H].tobytes())

        for layer in range(L):
            p = f"model.layers.{layer}"
            f.write(T(f"{p}.input_layernorm.weight").astype(np.float32).reshape(-1)[:H].tobytes())
            f.write(
                T(f"{p}.post_attention_layernorm.weight")
                .astype(np.float32)
                .reshape(-1)[:H]
                .tobytes()
            )
            f.write(fit2d(T(f"{p}.self_attn.q_proj.weight"), q_dim, H).tobytes())
            f.write(fit2d(T(f"{p}.self_attn.k_proj.weight"), kv_dim, H).tobytes())
            f.write(fit2d(T(f"{p}.self_attn.v_proj.weight"), kv_dim, H).tobytes())
            f.write(fit2d(T(f"{p}.self_attn.o_proj.weight"), H, q_dim).tobytes())
            f.write(fit2d(T(f"{p}.block_sparse_moe.gate.weight"), E, H).tobytes())
            print(f"  dense {layer}/{L - 1}", flush=True)

        for layer in range(L):
            for e in range(E):
                base = f"model.layers.{layer}.block_sparse_moe.experts.{e}"
                g = fit2d(T(f"{base}.w1.weight"), inter, H)
                u = fit2d(T(f"{base}.w3.weight"), inter, H)
                d = fit2d(T(f"{base}.w2.weight"), H, inter)
                blob = pack_expert(g, u, d, quant)
                if len(blob) < expert_size:
                    blob += b"\0" * (expert_size - len(blob))
                elif len(blob) > expert_size:
                    raise SystemExit(
                        f"expert blob too large: {len(blob)} > {expert_size} "
                        f"(L={layer} e={e})"
                    )
                f.write(blob)
                del g, u, d, blob
            print(f"  experts {layer}/{L - 1}", flush=True)

    shards.close()
    print(f"[done] {out}  {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
