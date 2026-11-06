#!/usr/bin/env python3
"""Create a tiny DeepSeek-style HF MoE fixture (routed + shared experts).

  H=64, L=2, heads=4, kv_heads=2, head_dim=16
  E=4 routed, S=1 shared, top_k=2, inter=32, V=256

Tensor names:
  model.layers.{L}.mlp.gate.weight
  model.layers.{L}.mlp.experts.{E}.gate_proj|up_proj|down_proj.weight
  model.layers.{L}.mlp.shared_experts.gate_proj|up_proj|down_proj.weight
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="models/hf_deepseek_moe_micro")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    H, L, E, S, V = 64, 2, 4, 1, 256
    n_heads, n_kv, hd = 4, 2, 16
    inter = 32
    assert n_heads * hd == H

    config = {
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "hidden_size": H,
        "intermediate_size": inter * 4,
        "moe_intermediate_size": inter,
        "num_hidden_layers": L,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": hd,
        "n_routed_experts": E,
        "n_shared_experts": S,
        "num_experts_per_tok": 2,
        "vocab_size": V,
        "rms_norm_eps": 1e-6,
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

        sp = f"{p}.mlp.shared_experts"
        tensors[f"{sp}.gate_proj.weight"] = rng.normal(
            0, 0.02, (inter, H)
        ).astype(np.float32)
        tensors[f"{sp}.up_proj.weight"] = rng.normal(
            0, 0.02, (inter, H)
        ).astype(np.float32)
        tensors[f"{sp}.down_proj.weight"] = rng.normal(
            0, 0.02, (H, inter)
        ).astype(np.float32)

    np.savez(out / "model.npz", **tensors)
    print(
        f"[deepseek-moe-micro] {out}  tensors={len(tensors)}  "
        f"H={H} L={L} E={E} S={S}"
    )

    try:
        from safetensors.numpy import save_file

        save_file(tensors, str(out / "model.safetensors"))
        print(f"[deepseek-moe-micro] wrote model.safetensors ({len(tensors)} tensors)")
    except ImportError:
        print(
            "[deepseek-moe-micro] safetensors not installed — skip .safetensors "
            "(stream path needs it)"
        )


if __name__ == "__main__":
    main()
