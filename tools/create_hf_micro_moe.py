#!/usr/bin/env python3
"""Create HF-like micro MoE with per-layer experts (Qwen3-MoE naming)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="models/hf_micro_moe")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    n_layers = 2
    hidden = 64
    vocab = 256
    n_experts = 4
    top_k = 2
    inter = 128
    n_heads = 4

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "vocab_size": vocab,
        "num_experts": n_experts,
        "num_local_experts": n_experts,
        "num_experts_per_tok": top_k,
        "moe_intermediate_size": inter,
        "intermediate_size": inter,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "rabbit_fixture": True,
        "rabbit_seed": args.seed,
        "rabbit_experts": "per-layer",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    tensors = {}
    tensors["model.embed_tokens.weight"] = (0.02 * rng.standard_normal((vocab, hidden))).astype(np.float32)
    tensors["lm_head.weight"] = (0.02 * rng.standard_normal((vocab, hidden))).astype(np.float32)
    tensors["model.norm.weight"] = (1.0 + 0.01 * rng.standard_normal(hidden)).astype(np.float32)

    for L in range(n_layers):
        p = f"model.layers.{L}"
        tensors[f"{p}.input_layernorm.weight"] = (1.0 + 0.01 * rng.standard_normal(hidden)).astype(np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = (
            1.0 + 0.01 * rng.standard_normal(hidden)
        ).astype(np.float32)

        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"{p}.self_attn.{name}.weight"] = (
                0.02 * rng.standard_normal((hidden, hidden))
            ).astype(np.float32)

        w_router = (0.05 * rng.standard_normal((n_experts, hidden))).astype(np.float32)
        w_router += np.arange(1, n_experts + 1, dtype=np.float32)[:, None] * 0.01
        tensors[f"{p}.mlp.gate.weight"] = w_router

        for e in range(n_experts):
            ep = f"{p}.mlp.experts.{e}"
            gate = np.zeros((inter, hidden), dtype=np.float32)
            up = np.zeros((inter, hidden), dtype=np.float32)
            down = np.zeros((hidden, inter), dtype=np.float32)
            for i in range(inter):
                for j in range(hidden):
                    gate[i, j] = 0.05 * np.sin(0.07 * (e + 1) * (i + 1) * (j + 1) * (L + 1))
                    up[i, j] = 0.05 * np.cos(0.05 * (e + 1) * (i + 2) * (j + 1) * (L + 1))
            for i in range(hidden):
                for j in range(inter):
                    down[i, j] = 0.04 * np.sin(0.03 * (e + 1) * (i + 1) * (j + 3) * (L + 1))
            gate += 0.001 * rng.standard_normal((inter, hidden), dtype=np.float32)
            up += 0.001 * rng.standard_normal((inter, hidden), dtype=np.float32)
            down += 0.001 * rng.standard_normal((hidden, inter), dtype=np.float32)
            tensors[f"{ep}.gate_proj.weight"] = gate
            tensors[f"{ep}.up_proj.weight"] = up
            tensors[f"{ep}.down_proj.weight"] = down

    np.savez_compressed(out / "model.npz", **tensors)

    try:
        from safetensors.numpy import save_file

        save_file(tensors, str(out / "model.safetensors"))
        print(f"wrote {out}/model.safetensors")
    except Exception:
        print("(safetensors not installed — using model.npz only)")

    print(f"wrote {out}/config.json and model.npz")
    print(f"  L={n_layers} E/layer={n_experts} records={n_layers * n_experts}")


if __name__ == "__main__":
    main()
