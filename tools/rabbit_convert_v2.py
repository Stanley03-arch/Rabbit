#!/usr/bin/env python3
"""v2 dense/expert packing and convert_hf."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from rabbit_convert_pack import (
    HEADER_SIZE,
    RABBIT_MAGIC,
    write_header,
    pack_with_scale,
    pack_int4_grouped,
    collect_tensors,
    classify_tensors,
    collect_experts_per_layer,
    pack_dense,
    pack_one_expert,
    _find,
    _find_layer,
)


def pack_v2_dense(tensors: Dict[str, np.ndarray], n_layers: int, hidden: int,
                  vocab: int, n_experts: int,
                  n_heads: int = 0, n_kv_heads: int = 0, head_dim: int = 0) -> bytes:
    embed = _find(tensors, "embed")
    if embed is None:
        embed = _find(tensors, "wte")
    if embed is None:
        raise RuntimeError("v2 layout: missing embedding weights")
    if embed.ndim != 2:
        raise RuntimeError(f"embed shape {embed.shape}")
    if embed.shape[0] == hidden and embed.shape[1] == vocab:
        embed = embed.T
    if embed.shape != (vocab, hidden):
        e = np.zeros((vocab, hidden), dtype=np.float32)
        v = min(vocab, embed.shape[0])
        h = min(hidden, embed.shape[1])
        e[:v, :h] = embed[:v, :h]
        embed = e

    lm = _find(tensors, "lm_head")
    if lm is None:
        lm = embed.copy()
    lm = np.asarray(lm, dtype=np.float32)
    if lm.shape == (vocab, hidden):
        lm = lm.T
    if lm.shape != (hidden, vocab):
        m = np.zeros((hidden, vocab), dtype=np.float32)
        h = min(hidden, lm.shape[0] if lm.ndim == 2 else 0)
        v = min(vocab, lm.shape[1] if lm.ndim == 2 else 0)
        if lm.ndim == 2:
            m[:h, :v] = lm[:h, :v]
        lm = m

    parts: List[bytes] = [embed.astype(np.float32).tobytes(), lm.astype(np.float32).tobytes()]

    for L in range(n_layers):
        rms_attn = _find_layer(tensors, L, "input_layernorm")
        if rms_attn is None:
            rms_attn = _find_layer(tensors, L, "attention_norm")
        if rms_attn is None:
            rms_attn = np.ones(hidden, dtype=np.float32)
        rms_attn = rms_attn.reshape(-1)[:hidden]
        if rms_attn.size < hidden:
            rms_attn = np.pad(rms_attn, (0, hidden - rms_attn.size), constant_values=1.0)

        rms_ffn = _find_layer(tensors, L, "post_attention_layernorm")
        if rms_ffn is None:
            rms_ffn = _find_layer(tensors, L, "ffn_norm")
        if rms_ffn is None:
            rms_ffn = np.ones(hidden, dtype=np.float32)
        rms_ffn = rms_ffn.reshape(-1)[:hidden]
        if rms_ffn.size < hidden:
            rms_ffn = np.pad(rms_ffn, (0, hidden - rms_ffn.size), constant_values=1.0)

        q = _find_layer(tensors, L, "q_proj")
        k = _find_layer(tensors, L, "k_proj")
        v = _find_layer(tensors, L, "v_proj")
        o = _find_layer(tensors, L, "o_proj")
        if q is None or k is None or v is None:
            qkv = _find_layer(tensors, L, "qkv")
            if qkv is not None and qkv.shape[0] >= 3 * hidden:
                q = qkv[:hidden]
                k = qkv[hidden : 2 * hidden]
                v = qkv[2 * hidden : 3 * hidden]
            else:
                q = k = v = np.eye(hidden, dtype=np.float32) * 0.02
        if o is None:
            o = np.eye(hidden, dtype=np.float32) * 0.02

        if n_heads <= 0:
            n_heads = 4 if hidden % 4 == 0 else 1
        if head_dim <= 0:
            head_dim = hidden // n_heads
        if n_kv_heads <= 0:
            n_kv_heads = n_heads
        q_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim

        def fit2d(m: np.ndarray, rows: int, cols: int) -> np.ndarray:
            m = np.asarray(m, dtype=np.float32)
            out = np.zeros((rows, cols), dtype=np.float32)
            if m.ndim != 2:
                return out
            r = min(rows, m.shape[0])
            c = min(cols, m.shape[1])
            out[:r, :c] = m[:r, :c]
            return out

        q = fit2d(q, q_dim, hidden)
        k = fit2d(k, kv_dim, hidden)
        v = fit2d(v, kv_dim, hidden)
        o = fit2d(o, hidden, q_dim)

        router = _find_layer(tensors, L, "block_sparse_moe.gate")
        if router is None:
            router = _find_layer(tensors, L, "mlp.gate")
        if router is None:
            router = _find_layer(tensors, L, "router")
        if router is None:
            router = 0.05 * np.random.randn(n_experts, hidden).astype(np.float32)
        router = np.asarray(router, dtype=np.float32)
        if router.ndim == 1:
            router = router.reshape(n_experts, -1)
        if router.shape[0] == hidden and router.shape[1] == n_experts:
            router = router.T
        rr = np.zeros((n_experts, hidden), dtype=np.float32)
        r0 = min(n_experts, router.shape[0])
        c0 = min(hidden, router.shape[1])
        rr[:r0, :c0] = router[:r0, :c0]

        parts.extend(
            [
                rms_attn.astype(np.float32).tobytes(),
                rms_ffn.astype(np.float32).tobytes(),
                q.astype(np.float32).tobytes(),
                k.astype(np.float32).tobytes(),
                v.astype(np.float32).tobytes(),
                o.astype(np.float32).tobytes(),
                rr.astype(np.float32).tobytes(),
            ]
        )

    return b"".join(parts)


def pack_v2_expert(expert_tensors: Dict[str, np.ndarray], quant: int,
                   hidden: int, inter: int) -> bytes:
    """Exact gate [I,H], up [I,H], down [H,I]. quant=0 → pure f32.

    Naming:
      HF / Qwen-style: gate_proj, up_proj, down_proj
      Mixtral-style:   w1=gate, w3=up, w2=down
    """
    gate = up = down = None
    for name, arr in expert_tensors.items():
        n = name.lower().replace("/", ".")
        a = np.asarray(arr, dtype=np.float32)
        if "gate_proj" in n or (n.endswith(".gate") and "moe" not in n):
            gate = a
        elif "up_proj" in n:
            up = a
        elif "down_proj" in n:
            down = a
        elif n.endswith(".w1") or ".w1." in n or n.endswith("w1.weight"):
            gate = a
        elif n.endswith(".w3") or ".w3." in n or n.endswith("w3.weight"):
            up = a
        elif n.endswith(".w2") or ".w2." in n or n.endswith("w2.weight"):
            down = a
        elif "gate" in n and "proj" in n:
            gate = a
        elif "up" in n and "gate" not in n:
            up = a
        elif "down" in n:
            down = a

    def fit(a: Optional[np.ndarray], rows: int, cols: int) -> np.ndarray:
        out = np.zeros((rows, cols), dtype=np.float32)
        if a is None:
            return out
        a = np.asarray(a, dtype=np.float32)
        if a.ndim != 2:
            a = a.reshape(rows, -1) if a.size >= rows else a.reshape(-1, cols)
        r = min(rows, a.shape[0])
        c = min(cols, a.shape[1])
        out[:r, :c] = a[:r, :c]
        return out

    gate = fit(gate, inter, hidden)
    up = fit(up, inter, hidden)
    down = fit(down, hidden, inter)
    if quant == 0:
        return (
            gate.astype(np.float32).tobytes()
            + up.astype(np.float32).tobytes()
            + down.astype(np.float32).tobytes()
        )
    if quant == 4:
        return pack_int4_grouped(gate) + pack_int4_grouped(up) + pack_int4_grouped(down)
    return pack_with_scale(gate, quant) + pack_with_scale(up, quant) + pack_with_scale(down, quant)


def load_hf_config_local(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        return json.load(f)


def convert_hf(model_dir: Path, out_path: Path, max_experts: Optional[int],
               quant: int, layout: str) -> None:
    config = load_hf_config_local(model_dir)
    print("[config]", {k: config.get(k) for k in (
        "model_type", "hidden_size", "num_hidden_layers", "num_local_experts",
        "num_experts_per_tok", "vocab_size", "moe_intermediate_size",
        "n_routed_experts", "num_experts", "num_attention_heads",
        "num_key_value_heads", "head_dim", "intermediate_size",
    ) if k in config})

    tensors = collect_tensors(model_dir)
    print(f"[info] {len(tensors)} tensors")

    dense, experts = classify_tensors(tensors)
    print(f"[info] dense={len(dense)}  expert_groups={len(experts)}")

    hidden_dim = int(config.get("hidden_size") or config.get("n_embd") or 1024)
    n_layers = int(config.get("num_hidden_layers") or config.get("n_layer") or 12)
    vocab_size = int(config.get("vocab_size") or 32000)
    experts_per_token = int(config.get("num_experts_per_tok") or config.get("moe_top_k") or 2)
    expert_dim = int(config.get("moe_intermediate_size") or config.get("intermediate_size") or hidden_dim * 2)

    n_heads = int(config.get("num_attention_heads") or 0)
    n_kv_heads = int(config.get("num_key_value_heads") or n_heads or 0)
    head_dim = int(config.get("head_dim") or (hidden_dim // n_heads if n_heads else 0))

    expert_ids = sorted(experts.keys())
    if max_experts is not None and len(expert_ids) > max_experts:
        print(f"[info] limiting experts {len(expert_ids)} → {max_experts}")
        expert_ids = expert_ids[:max_experts]

    n_experts = len(expert_ids) if expert_ids else int(
        config.get("num_local_experts") or config.get("num_experts") or config.get("n_routed_experts") or 8
    )

    if layout == "v2":
        dense_blob = pack_v2_dense(
            tensors, n_layers, hidden_dim, vocab_size, n_experts,
            n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim,
        )
        per_layer = collect_experts_per_layer(tensors, n_layers, n_experts)
        expert_blobs = []
        for L in range(n_layers):
            for e in range(n_experts):
                expert_blobs.append(
                    pack_v2_expert(
                        per_layer[L][e],
                        quant=quant,
                        hidden=hidden_dim,
                        inter=expert_dim,
                    )
                )
        if not expert_blobs:
            raise RuntimeError("v2 layout: no expert tensors found")
        expert_size = max(len(b) for b in expert_blobs)
        expert_blobs = [b + b"\0" * (expert_size - len(b)) for b in expert_blobs]
        dense_quant = 0
        name = model_dir.name[:60]
        n_records = n_layers * n_experts
    else:
        dense_blob = pack_dense(dense, quant=8)
        expert_blobs = []
        for eid in expert_ids:
            expert_blobs.append(
                pack_one_expert(experts[eid], quant=quant, layout=layout, hidden_dim=hidden_dim)
            )
        if not expert_blobs:
            raise RuntimeError("no expert tensors found")
        expert_size = max(len(b) for b in expert_blobs)
        expert_blobs = [b + b"\0" * (expert_size - len(b)) for b in expert_blobs]
        dense_quant = 8
        name = model_dir.name[:60]
        n_records = n_experts

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        write_header(
            f,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            n_experts=n_experts,
            experts_per_token=experts_per_token,
            expert_dim=expert_dim,
            dense_size=len(dense_blob),
            expert_size=expert_size,
            dense_quant=dense_quant,
            expert_quant=quant,
            name=name,
            n_expert_records=n_records,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
        )
        f.write(dense_blob)
        for b in expert_blobs:
            f.write(b)

    total = out_path.stat().st_size
    print(f"[done] {out_path}")
    print(f"       dense={len(dense_blob)/1024:.1f} KB  experts={n_experts} x {expert_size/1024:.1f} KB")
    print(f"       quant={quant}  layout={layout}  total={total/1024:.1f} KB")
    if n_heads:
        print(f"       GQA: heads={n_heads} kv_heads={n_kv_heads} head_dim={head_dim}")
    if layout == "v2":
        print(f"       forward: rabbit ref-v2 (./rabbit chat --model {out_path})")
        print(f"       expert records: {n_layers * n_experts} (per-layer)")
    else:
        print(f"\nRun:  ./rabbit chat --model {out_path}")
