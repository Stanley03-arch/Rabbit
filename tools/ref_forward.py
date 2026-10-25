#!/usr/bin/env python3
"""NumPy reference forward for rabbit-ref-v1 (MoE-only parity core)."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

HEADER_FMT = "<8sIIIIIIIIQQQII64s128s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


def load_ref(path: Path):
    data = path.read_bytes()
    hdr = struct.unpack_from(HEADER_FMT, data, 0)
    (
        magic,
        version,
        header_size,
        n_layers,
        hidden_dim,
        vocab_size,
        n_experts,
        top_k,
        expert_dim,
        dense_size,
        expert_size,
        total_experts_size,
        dense_quant,
        expert_quant,
        name,
        _reserved,
    ) = hdr
    assert magic == b"RABBIT01", magic
    off = HEADER_SIZE
    dense = data[off : off + dense_size]
    off += dense_size
    experts = []
    for e in range(n_experts):
        blob = data[off : off + expert_size]
        off += expert_size
        scale = struct.unpack_from("<f", blob, 0)[0]
        payload = blob[4:]
        # int8 matrix HxH
        need = hidden_dim * hidden_dim
        W = np.frombuffer(payload[:need], dtype=np.int8).reshape(hidden_dim, hidden_dim).astype(np.float32)
        experts.append((scale, W))
    return {
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "n_experts": n_experts,
        "top_k": top_k,
        "experts": experts,
        "name": name.split(b"\0", 1)[0].decode(),
    }


def top_k_route(h: np.ndarray, n_experts: int, k: int):
    scores = np.zeros(n_experts, dtype=np.float32)
    H = h.shape[0]
    for e in range(n_experts):
        s = 0.0
        for i in range(min(32, H)):
            s += float(h[i]) * float((e + 1) * (i + 3))
        scores[e] = s
    idx = np.argsort(-scores)[:k]
    logits = scores[idx]
    logits = logits - logits.max()
    w = np.exp(logits)
    w = w / w.sum()
    return idx.tolist(), w.astype(np.float32)


def moe_step(model, h: np.ndarray) -> np.ndarray:
    h = h.copy()
    ids, weights = top_k_route(h, model["n_experts"], model["top_k"])
    for eid, w in zip(ids, weights):
        scale, W = model["experts"][eid]
        # y += 0.05 * w * scale * (W @ h)
        h = h + 0.05 * float(w) * float(scale) * (W @ h)
    return h


def forward(model, prompt: str) -> dict:
    H = model["hidden_dim"]
    h = np.zeros(H, dtype=np.float32)
    for i, ch in enumerate(prompt.encode("utf-8", errors="replace")):
        h[i % H] += (float(ch % 64) - 32.0) / 32.0

    for _ in range(model["n_layers"]):
        h = moe_step(model, h)

    return {
        "sum": float(h.sum()),
        "l1": float(np.abs(h).sum()),
        "l2": float(np.sqrt((h * h).sum())),
        "first8": [float(x) for x in h[:8]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ref-v1.rabbit")
    ap.add_argument("--prompt", default="hi")
    args = ap.parse_args()
    model = load_ref(Path(args.model))
    stats = forward(model, args.prompt)
    print("REF", args.prompt, stats)


if __name__ == "__main__":
    main()
