#!/usr/bin/env python3
"""Create a tiny Mixtral-named HF-style fixture for Rabbit conversion tests.

Shapes mirror TinyBrainBot / Mixtral but scaled down:
  H=64, L=2, heads=4, kv_heads=2, head_dim=16, E=4, top_k=2, inter=32, V=256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="models/hf_mixtral_micro")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    H, L, E, V = 64, 2, 4, 256
    n_heads, n_kv, hd = 4, 2, 16
    inter = 32
    assert n_heads * hd == H

    config = {
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "hidden_size": H,
        "intermediate_size": inter,
        "num_hidden_layers": L,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": hd,
        "num_local_experts": E,
        "num_experts_per_tok": 2,
        "vocab_size": V,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "hidden_act": "silu",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    tensors = {}
    tensors["model.embed_tokens.weight"] = rng.normal(0, 0.02, (V, H)).astype(np.float32)
    tensors["model.norm.weight"] = np.ones(H, dtype=np.float32)

    for layer in range(L):
        p = f"model.layers.{layer}"
        tensors[f"{p}.input_layernorm.weight"] = np.ones(H, dtype=np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones(H, dtype=np.float32)
        tensors[f"{p}.self_attn.q_proj.weight"] = rng.normal(0, 0.02, (n_heads * hd, H)).astype(np.float32)
        tensors[f"{p}.self_attn.k_proj.weight"] = rng.normal(0, 0.02, (n_kv * hd, H)).astype(np.float32)
        tensors[f"{p}.self_attn.v_proj.weight"] = rng.normal(0, 0.02, (n_kv * hd, H)).astype(np.float32)
        tensors[f"{p}.self_attn.o_proj.weight"] = rng.normal(0, 0.02, (H, n_heads * hd)).astype(np.float32)
        tensors[f"{p}.block_sparse_moe.gate.weight"] = rng.normal(0, 0.02, (E, H)).astype(np.float32)
        for e in range(E):
            ep = f"{p}.block_sparse_moe.experts.{e}"
            tensors[f"{ep}.w1.weight"] = rng.normal(0, 0.02, (inter, H)).astype(np.float32)
            tensors[f"{ep}.w3.weight"] = rng.normal(0, 0.02, (inter, H)).astype(np.float32)
            tensors[f"{ep}.w2.weight"] = rng.normal(0, 0.02, (H, inter)).astype(np.float32)

    np.savez(out / "model.npz", **tensors)
    print(f"[mixtral-micro] {out}  tensors={len(tensors)}  H={H} L={L} E={E}")


if __name__ == "__main__":
    main()
