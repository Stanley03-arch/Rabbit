#!/usr/bin/env python3
"""Create a faithful rabbit-ref-v2 model for C ↔ NumPy parity."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

RABBIT_MAGIC = b"RABBIT01"
HEADER_SIZE = struct.calcsize("8s" + "I" * 8 + "Q" * 3 + "I" * 2 + "64s" + "128s")


def write_header(f, **kw) -> None:
    name_bytes = kw["name"].encode("utf-8")[:63].ljust(64, b"\0")
    reserved = bytearray(128)
    n_heads = int(kw.get("n_heads") or 0)
    n_kv_heads = int(kw.get("n_kv_heads") or 0)
    head_dim = int(kw.get("head_dim") or 0)
    if n_heads and head_dim:
        struct.pack_into("<III", reserved, 0, n_heads, n_kv_heads or n_heads, head_dim)
    n_records = kw.get("n_expert_records", kw["n_experts"])
    total_experts_size = n_records * kw["expert_size"]
    header = struct.pack(
        "<8sIIIIIIIIQQQII64s128s",
        RABBIT_MAGIC,
        1,
        HEADER_SIZE,
        kw["n_layers"],
        kw["hidden_dim"],
        kw["vocab_size"],
        kw["n_experts"],
        kw["experts_per_token"],
        kw["expert_dim"],
        kw["dense_size"],
        kw["expert_size"],
        total_experts_size,
        kw["dense_quant"],
        kw["expert_quant"],
        name_bytes,
        bytes(reserved),
    )
    f.write(header)


def make_expert(rng, e, L, inter, hidden):
    gate = (0.02 * rng.standard_normal((inter, hidden))).astype(np.float32)
    up = (0.02 * rng.standard_normal((inter, hidden))).astype(np.float32)
    down = (0.02 * rng.standard_normal((hidden, inter))).astype(np.float32)
    gate += 0.001 * (e + 1) + 0.0001 * L
    up += 0.001 * (e + 1)
    down += 0.001 * (e + 1)
    return gate.tobytes() + up.tobytes() + down.tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_layers = 2
    hidden = 64
    vocab = 256
    n_experts = 4
    top_k = 2
    inter = 128
    n_heads = 4
    head_dim = hidden // n_heads

    embed = (0.02 * rng.standard_normal((vocab, hidden))).astype(np.float32)
    lm_head = (0.02 * rng.standard_normal((hidden, vocab))).astype(np.float32)

    final_norm = (1.0 + 0.01 * rng.standard_normal(hidden)).astype(np.float32)
    dense_parts = [embed.tobytes(), lm_head.tobytes(), final_norm.tobytes()]
    for L in range(n_layers):
        rms_attn = (1.0 + 0.01 * rng.standard_normal(hidden)).astype(np.float32)
        rms_ffn = (1.0 + 0.01 * rng.standard_normal(hidden)).astype(np.float32)
        w_qkv = (0.02 * rng.standard_normal((3 * hidden, hidden))).astype(np.float32)
        w_o = (0.02 * rng.standard_normal((hidden, hidden))).astype(np.float32)
        w_router = (0.05 * rng.standard_normal((n_experts, hidden))).astype(np.float32)
        w_router += np.arange(1, n_experts + 1, dtype=np.float32)[:, None] * 0.01
        dense_parts.extend(
            [
                rms_attn.tobytes(),
                rms_ffn.tobytes(),
                w_qkv.tobytes(),
                w_o.tobytes(),
                w_router.tobytes(),
            ]
        )

    dense_blob = b"".join(dense_parts)

    expert_blobs = []
    for L in range(n_layers):
        for e in range(n_experts):
            expert_blobs.append(make_expert(rng, e, L, inter, hidden))

    expert_size = max(len(b) for b in expert_blobs)
    expert_blobs = [b + b"\0" * (expert_size - len(b)) for b in expert_blobs]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        write_header(
            f,
            n_layers=n_layers,
            hidden_dim=hidden,
            vocab_size=vocab,
            n_experts=n_experts,
            experts_per_token=top_k,
            expert_dim=inter,
            dense_size=len(dense_blob),
            expert_size=expert_size,
            dense_quant=0,
            expert_quant=0,
            name="ref-v2",
            n_expert_records=n_layers * n_experts,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            head_dim=head_dim,
        )
        f.write(dense_blob)
        for b in expert_blobs:
            f.write(b)

    print(f"wrote {out} ({out.stat().st_size} bytes) seed={args.seed} ref-v2")
    print(f"  layers={n_layers} H={hidden} E/layer={n_experts} records={n_layers * n_experts}")


if __name__ == "__main__":
    main()
