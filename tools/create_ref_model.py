#!/usr/bin/env python3
"""Create deterministic rabbit-ref-v1 model (seed=1)."""

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
    total_experts = kw["n_experts"] * kw["expert_size"]
    f.write(
        struct.pack(
            HEADER_FMT,
            MAGIC,
            1,
            struct.calcsize(HEADER_FMT),
            kw["n_layers"],
            kw["hidden_dim"],
            kw["vocab_size"],
            kw["n_experts"],
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


def pack_int8_with_scale(mat: np.ndarray) -> bytes:
    f = mat.astype(np.float32).ravel()
    scale = float(np.max(np.abs(f)) / 127.0 + 1e-8)
    q = np.clip(np.round(f / scale), -128, 127).astype(np.int8)
    return struct.pack("<f", scale) + q.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/ref-v1.rabbit")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    n_layers = 2
    hidden = 32
    vocab = 64
    n_experts = 4
    top_k = 2

    # Dense: unused by MoE parity core but required by format
    dense = rng.standard_normal(hidden * 64, dtype=np.float32)
    d_scale = float(np.max(np.abs(dense)) / 127.0 + 1e-8)
    d_q = np.clip(np.round(dense / d_scale), -128, 127).astype(np.int8)
    dense_blob = struct.pack("<f", d_scale) + d_q.tobytes()

    expert_blobs = []
    for e in range(n_experts):
        # Fixed pattern + small noise for determinism across versions
        mat = np.zeros((hidden, hidden), dtype=np.float32)
        for o in range(hidden):
            for i in range(hidden):
                mat[o, i] = ((e + 1) * 0.01) * np.sin(0.1 * (o + 1) * (i + 1))
        mat += 0.001 * rng.standard_normal((hidden, hidden), dtype=np.float32)
        expert_blobs.append(pack_int8_with_scale(mat))

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
            expert_dim=hidden * 2,
            dense_size=len(dense_blob),
            expert_size=expert_size,
            dense_quant=8,
            expert_quant=8,
            name="ref-v1",
        )
        f.write(dense_blob)
        for b in expert_blobs:
            f.write(b)

    print(f"wrote {out} ({out.stat().st_size} bytes) seed={args.seed}")


if __name__ == "__main__":
    main()
