#!/usr/bin/env python3
"""Low-RAM streaming HF safetensors → rabbit-ref-v2 (group-int4 experts).

Uses exact Mixtral tensor names + numpy-vectorized int4 packing.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rabbit_convert_pack import write_header


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
    r, c = min(rows, m.shape[0]), min(cols, m.shape[1])
    out[:r, :c] = m[:r, :c]
    return out


def expert_nbytes(inter, H, gs=32):
    def one(n):
        ng = (n + gs - 1) // gs
        return 4 + ng * 4 + (n + 1) // 2
    return one(inter * H) * 2 + one(H * inter)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", type=int, default=4, choices=[4])
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    H = int(config["hidden_size"])
    L = int(config["num_hidden_layers"])
    V = int(config["vocab_size"])
    E = int(config.get("num_local_experts") or config.get("num_experts"))
    top_k = int(config.get("num_experts_per_tok") or 2)
    inter = int(config.get("moe_intermediate_size") or config["intermediate_size"])
    n_heads = int(config["num_attention_heads"])
    n_kv = int(config.get("num_key_value_heads") or n_heads)
    head_dim = int(config.get("head_dim") or H // n_heads)
    q_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim

    expert_size = expert_nbytes(inter, H)
    dense_size = (V * H + H * V + H) * 4
    dense_size += L * (H * 2 + q_dim * H + kv_dim * H + kv_dim * H + H * q_dim + E * H) * 4
    print(f"[config] H={H} L={L} V={V} E={E} inter={inter} heads={n_heads}/{n_kv}/{head_dim}")
    print(f"[plan] dense={dense_size/1e6:.1f}MB expert={expert_size/1e3:.1f}KB records={L*E}")

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        raise SystemExit(f"No safetensors in {model_dir}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with safe_open(str(st_files[0]), framework="np") as st:
        def T(name):
            return np.asarray(st.get_tensor(name), dtype=np.float32)

        with open(out, "wb") as f:
            write_header(
                f, n_layers=L, hidden_dim=H, vocab_size=V, n_experts=E,
                experts_per_token=top_k, expert_dim=inter, dense_size=dense_size,
                expert_size=expert_size, dense_quant=0, expert_quant=4,
                name=model_dir.name[:60], n_expert_records=L * E,
                n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim,
            )
            embed = T("model.embed_tokens.weight")
            emb = fit2d(embed, V, H)
            f.write(emb.tobytes())
            f.write(emb.T.copy().tobytes())  # tied lm_head
            del embed, emb
            f.write(T("model.norm.weight").astype(np.float32).tobytes())

            for layer in range(L):
                p = f"model.layers.{layer}"
                f.write(T(f"{p}.input_layernorm.weight").astype(np.float32).tobytes())
                f.write(T(f"{p}.post_attention_layernorm.weight").astype(np.float32).tobytes())
                f.write(fit2d(T(f"{p}.self_attn.q_proj.weight"), q_dim, H).tobytes())
                f.write(fit2d(T(f"{p}.self_attn.k_proj.weight"), kv_dim, H).tobytes())
                f.write(fit2d(T(f"{p}.self_attn.v_proj.weight"), kv_dim, H).tobytes())
                f.write(fit2d(T(f"{p}.self_attn.o_proj.weight"), H, q_dim).tobytes())
                f.write(fit2d(T(f"{p}.block_sparse_moe.gate.weight"), E, H).tobytes())
                print(f"  dense {layer}/{L-1}", flush=True)

            for layer in range(L):
                for e in range(E):
                    base = f"model.layers.{layer}.block_sparse_moe.experts.{e}"
                    g = fit2d(T(f"{base}.w1.weight"), inter, H)
                    u = fit2d(T(f"{base}.w3.weight"), inter, H)
                    d = fit2d(T(f"{base}.w2.weight"), H, inter)
                    blob = pack_int4_grouped_fast(g) + pack_int4_grouped_fast(u) + pack_int4_grouped_fast(d)
                    if len(blob) < expert_size:
                        blob += b"\0" * (expert_size - len(blob))
                    f.write(blob)
                    del g, u, d, blob
                print(f"  experts {layer}/{L-1}", flush=True)

    print(f"[done] {out}  {out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
