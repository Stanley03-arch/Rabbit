#!/usr/bin/env python3
"""Create deterministic rabbit-ref-v2 model (seed=2) — per-layer experts."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

MAGIC = b"RABBIT01"
HEADER_FMT = "<8sIIIIIIIIQQQII64s128s"


def write_header(f, **kw):
    name = kw["name"].encode("utf-8")[:63].ljust(64, b"\0")
    reserved = b"\0" * 128
    total_experts = kw["n_expert_records"] * kw["expert_size"]
    f.write(
        struct.pack(
            HEADER_FMT,
            MAGIC,
            1,
            struct.calcsize(HEADER_FMT),
            kw["n_layers"],
            kw["hidden_dim"],
            kw["vocab_size"],
            kw["n_experts"],  # per layer
            kw["experts_per_token"],
            kw["expert_dim"],
            kw["dense_size"],
            kw["expert_size"],
            total_experts,
            kw["dense_quant"],
            kw["expert_quant"],
            name,
            reserved,
        )
    )


def pack_int8(mat: np.ndarray) -> bytes:
    f = mat.astype(np.float32).ravel()
    scale = float(np.max(np.abs(f)) / 127.0 + 1e-8)
    q = np.clip(np.round(f / scale), -128, 127).astype(np.int8)
    return struct.pack("<f", scale) + q.tobytes()


def make_expert(rng, e, L, inter, hidden):
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
    return pack_int8(gate) + pack_int8(up) + pack_int8(down)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/ref-v2.rabbit")
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    n_layers = 2
    hidden = 64
    vocab = 256
    n_experts = 4
    top_k = 2
    inter = 128

    embed = (0.02 * rng.standard_normal((vocab, hidden))).astype(np.float32)
    lm_head = (0.02 * rng.standard_normal((hidden, vocab))).astype(np.float32)

    dense_parts = [embed.tobytes(), lm_head.tobytes()]
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

    # Per-layer experts: L0E0..L0E3, L1E0..
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
            n_expert_records=n_layers * n_experts,
            experts_per_token=top_k,
            expert_dim=inter,
            dense_size=len(dense_blob),
            expert_size=expert_size,
            dense_quant=0,
            expert_quant=8,
            name="ref-v2",
        )
        f.write(dense_blob)
        for b in expert_blobs:
            f.write(b)

    print(f"wrote {out} ({out.stat().st_size} bytes) seed={args.seed} ref-v2")
    print(f"  layers={n_layers} H={hidden} E/layer={n_experts} records={n_layers * n_experts}")


if __name__ == "__main__":
    main()
