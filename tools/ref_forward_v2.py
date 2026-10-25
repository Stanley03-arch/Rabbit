#!/usr/bin/env python3
"""NumPy oracle for rabbit-ref-v2 (faithful micro MoE)."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HEADER_FMT = "<8sIIIIIIIIQQQII64s128s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
EPS = 1e-5


def _read_f32(data: bytes, off: int, n: int) -> Tuple[np.ndarray, int]:
    arr = np.frombuffer(data, dtype=np.float32, count=n, offset=off).copy()
    return arr, off + n * 4


def _dequant_int8(blob: bytes, off: int, n: int) -> Tuple[np.ndarray, float, int]:
    scale = struct.unpack_from("<f", blob, off)[0]
    off += 4
    q = np.frombuffer(blob, dtype=np.int8, count=n, offset=off).astype(np.float32)
    off += n
    return q * scale, scale, off


def load_ref_v2(path: Path) -> Dict[str, Any]:
    data = path.read_bytes()
    hdr = struct.unpack_from(HEADER_FMT, data, 0)
    (
        magic,
        version,
        header_size,
        n_layers,
        hidden,
        vocab,
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

    d_off = 0
    embed, d_off = _read_f32(dense, d_off, vocab * hidden)
    embed = embed.reshape(vocab, hidden)
    lm_head, d_off = _read_f32(dense, d_off, hidden * vocab)
    lm_head = lm_head.reshape(hidden, vocab)

    layers = []
    for _ in range(n_layers):
        rms_attn, d_off = _read_f32(dense, d_off, hidden)
        rms_ffn, d_off = _read_f32(dense, d_off, hidden)
        w_qkv, d_off = _read_f32(dense, d_off, 3 * hidden * hidden)
        w_qkv = w_qkv.reshape(3 * hidden, hidden)
        w_o, d_off = _read_f32(dense, d_off, hidden * hidden)
        w_o = w_o.reshape(hidden, hidden)
        w_router, d_off = _read_f32(dense, d_off, n_experts * hidden)
        w_router = w_router.reshape(n_experts, hidden)
        layers.append(
            {
                "rms_attn": rms_attn,
                "rms_ffn": rms_ffn,
                "w_qkv": w_qkv,
                "w_o": w_o,
                "w_router": w_router,
            }
        )

    experts = []
    inter = expert_dim
    for e in range(n_experts):
        blob = data[off : off + expert_size]
        off += expert_size
        e_off = 0
        gate, _, e_off = _dequant_int8(blob, e_off, inter * hidden)
        gate = gate.reshape(inter, hidden)
        up, _, e_off = _dequant_int8(blob, e_off, inter * hidden)
        up = up.reshape(inter, hidden)
        down, _, e_off = _dequant_int8(blob, e_off, hidden * inter)
        down = down.reshape(hidden, inter)
        experts.append({"gate": gate, "up": up, "down": down})

    return {
        "n_layers": n_layers,
        "hidden": hidden,
        "vocab": vocab,
        "n_experts": n_experts,
        "top_k": top_k,
        "expert_dim": inter,
        "n_heads": 4,
        "head_dim": hidden // 4,
        "embed": embed,
        "lm_head": lm_head,
        "layers": layers,
        "experts": experts,
        "name": name.split(b"\0", 1)[0].decode(),
    }


def rmsnorm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    var = np.mean(x * x, axis=-1, keepdims=True)
    x = x / np.sqrt(var + EPS)
    return x * weight


def rope(q: np.ndarray, k: np.ndarray, pos: int, head_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    half = head_dim // 2
    freqs = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float32) / half))
    angles = pos * freqs
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    def apply(x):
        x = x.reshape(x.shape[0], half, 2)
        x0, x1 = x[..., 0], x[..., 1]
        y0 = x0 * cos_a - x1 * sin_a
        y1 = x0 * sin_a + x1 * cos_a
        return np.stack([y0, y1], axis=-1).reshape(x.shape[0], head_dim)

    return apply(q), apply(k)


def silu(x: np.ndarray) -> np.ndarray:
    return x * (1.0 / (1.0 + np.exp(-x)))


def attention(
    x: np.ndarray,
    layer: Dict[str, Any],
    kv: Dict[str, List[np.ndarray]],
    n_heads: int,
    head_dim: int,
    pos: int,
) -> np.ndarray:
    H = x.shape[0]
    x_n = rmsnorm(x, layer["rms_attn"])
    qkv = layer["w_qkv"] @ x_n
    q, k, v = np.split(qkv, 3)
    q = q.reshape(n_heads, head_dim)
    k = k.reshape(n_heads, head_dim)
    v = v.reshape(n_heads, head_dim)
    q, k = rope(q, k, pos, head_dim)

    kv["k"].append(k.copy())
    kv["v"].append(v.copy())

    scale = 1.0 / np.sqrt(head_dim)
    out = np.zeros_like(q)
    for h in range(n_heads):
        scores = []
        for t in range(len(kv["k"])):
            scores.append(float(np.dot(q[h], kv["k"][t][h]) * scale))
        scores_a = np.array(scores, dtype=np.float32)
        scores_a = scores_a - scores_a.max()
        w = np.exp(scores_a)
        w = w / w.sum()
        acc = np.zeros(head_dim, dtype=np.float32)
        for t, wt in enumerate(w):
            acc += wt * kv["v"][t][h]
        out[h] = acc

    y = out.reshape(H)
    y = layer["w_o"] @ y
    return x + y


def moe_swiglu(x: np.ndarray, layer: Dict[str, Any], experts: List[Dict], top_k: int) -> np.ndarray:
    x_n = rmsnorm(x, layer["rms_ffn"])
    logits = layer["w_router"] @ x_n
    idx = np.argsort(-logits)[:top_k]
    sel = logits[idx]
    sel = sel - sel.max()
    w = np.exp(sel)
    w = w / w.sum()

    acc = np.zeros_like(x)
    for eid, wt in zip(idx.tolist(), w.tolist()):
        e = experts[eid]
        g = silu(e["gate"] @ x_n)
        u = e["up"] @ x_n
        mid = g * u
        acc = acc + float(wt) * (e["down"] @ mid)
    return x + acc


def forward_logits(model: Dict[str, Any], token_ids: List[int]) -> np.ndarray:
    H = model["hidden"]
    n_heads = model["n_heads"]
    head_dim = model["head_dim"]
    # one KV cache per layer (matches C)
    kv_layers = [{"k": [], "v": []} for _ in model["layers"]]

    h = np.zeros(H, dtype=np.float32)
    for pos, tid in enumerate(token_ids):
        tid = int(tid) % model["vocab"]
        h = model["embed"][tid].copy()
        for li, layer in enumerate(model["layers"]):
            h = attention(h, layer, kv_layers[li], n_heads, head_dim, pos)
            h = moe_swiglu(h, layer, model["experts"], model["top_k"])

    logits = model["lm_head"].T @ h
    return logits


def greedy_generate(model: Dict[str, Any], prompt: str, n_new: int) -> List[int]:
    ids = [ord(c) % model["vocab"] for c in prompt]
    if not ids:
        ids = [1]
    for _ in range(n_new):
        logits = forward_logits(model, ids)
        nxt = int(np.argmax(logits))
        ids.append(nxt)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ref-v2.rabbit")
    ap.add_argument("--prompt", default="hi")
    ap.add_argument("--tokens", type=int, default=8)
    args = ap.parse_args()

    model = load_ref_v2(Path(args.model))
    ids = greedy_generate(model, args.prompt, args.tokens)
    print("REF_V2", args.prompt, "ids=", ids)
    print("last8", ids[-8:])
    logits = forward_logits(model, ids[:-1] if len(ids) > 1 else ids)
    print("argmax", int(np.argmax(logits)), "l2", float(np.linalg.norm(logits)))


if __name__ == "__main__":
    main()
