#!/usr/bin/env python3
"""Create a tiny Qwen2/3-MoE-named HF-style fixture for Rabbit conversion tests.

Same shapes as Mixtral micro, but tensor names follow Qwen MoE:
  model.layers.{L}.mlp.gate.weight
  model.layers.{L}.mlp.experts.{E}.gate_proj|up_proj|down_proj.weight

  H=64, L=2, heads=4, kv_heads=2, head_dim=16, E=4, top_k=2, inter=32, V=256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="models/hf_qwen_moe_micro")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    H, L, E, V = 64, 2, 4, 256
    n_heads, n_kv, hd = 4, 2, 16
    inter = 32
    assert n_heads * hd == H

    config = {
        "architectures": ["Qwen2MoeForCausalLM"],
        "model_type": "qwen2_moe",
        "hidden_size": H,
        "intermediate_size": inter,
        "moe_intermediate_size": inter,
        "num_hidden_layers": L,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": hd,
        "num_experts": E,
        "num_experts_per_tok": 2,
        "vocab_size": V,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "hidden_act": "silu",
        "norm_topk_prob": True,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    tensors = {}
    tensors["model.embed_tokens.weight"] = rng.normal(0, 0.02, (V, H)).astype(np.float32)
    tensors["model.norm.weight"] = np.ones(H, dtype=np.float32)

    for layer in range(L):
        p = f"model.layers.{layer}"
        tensors[f"{p}.input_layernorm.weight"] = np.ones(H, dtype=np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones(H, dtype=np.float32)
        tensors[f"{p}.self_attn.q_proj.weight"] = rng.normal(
            0, 0.02, (n_heads * hd, H)
        ).astype(np.float32)
        tensors[f"{p}.self_attn.k_proj.weight"] = rng.normal(
            0, 0.02, (n_kv * hd, H)
        ).astype(np.float32)
        tensors[f"{p}.self_attn.v_proj.weight"] = rng.normal(
            0, 0.02, (n_kv * hd, H)
        ).astype(np.float32)
        tensors[f"{p}.self_attn.o_proj.weight"] = rng.normal(
            0, 0.02, (H, n_heads * hd)
        ).astype(np.float32)
        tensors[f"{p}.mlp.gate.weight"] = rng.normal(0, 0.02, (E, H)).astype(np.float32)
        for e in range(E):
            ep = f"{p}.mlp.experts.{e}"
            tensors[f"{ep}.gate_proj.weight"] = rng.normal(
                0, 0.02, (inter, H)
            ).astype(np.float32)
            tensors[f"{ep}.up_proj.weight"] = rng.normal(
                0, 0.02, (inter, H)
            ).astype(np.float32)
            tensors[f"{ep}.down_proj.weight"] = rng.normal(
                0, 0.02, (H, inter)
            ).astype(np.float32)

    np.savez(out / "model.npz", **tensors)
    print(f"[qwen-moe-micro] {out}  tensors={len(tensors)}  H={H} L={L} E={E}")

    try:
        from safetensors.numpy import save_file

        save_file(tensors, str(out / "model.safetensors"))
        print(f"[qwen-moe-micro] wrote model.safetensors ({len(tensors)} tensors)")
    except ImportError:
        print(
            "[qwen-moe-micro] safetensors not installed — skip .safetensors "
            "(stream path needs it)"
        )


if __name__ == "__main__":
    main()
