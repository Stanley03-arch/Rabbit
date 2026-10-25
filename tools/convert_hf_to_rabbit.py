#!/usr/bin/env python3
"""
convert_hf_to_rabbit.py

Convert open MoE models into the Rabbit binary format.

Layouts:
  square — legacy HxH expert (lossy)
  flat   — gate/up/down ordered for SwiGLU kernel (still opaque dense)
  v2     — Phase D faithful dense + exact gate/up/down (no square crop)
           packs per-layer experts: n_expert_records = n_layers x n_experts

Expert quant:
  0 — pure float32 (exact parity vs unquantized weights)
  4 — group-scaled int4 (group=32)
  8 — per-tensor scaled int8

Usage:
  python tools/convert_hf_to_rabbit.py --demo --out models/demo-moe.rabbit
  python tools/convert_hf_to_rabbit.py --model-dir /path/to/hf --out models/m.rabbit --layout v2 --quant 4
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

RABBIT_MAGIC = b"RABBIT01"
HEADER_SIZE = struct.calcsize(
    "8s" + "I" * 8 + "Q" * 3 + "I" * 2 + "64s" + "128s"
)


def write_header(f, **kw) -> None:
    name_bytes = kw["name"].encode("utf-8")[:63].ljust(64, b"\0")
    reserved = b"\0" * 128
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
        reserved,
    )
    f.write(header)


def quant_int8(arr: np.ndarray) -> Tuple[bytes, float]:
    f = arr.astype(np.float32).ravel()
    scale = float(np.max(np.abs(f)) / 127.0 + 1e-8)
    q = np.clip(np.round(f / scale), -128, 127).astype(np.int8)
    return q.tobytes(), scale


def quant_int4(arr: np.ndarray) -> Tuple[bytes, float]:
    f = arr.astype(np.float32).ravel()
    scale = float(np.max(np.abs(f)) / 7.0 + 1e-8)
    q = np.clip(np.round(f / scale), -8, 7).astype(np.int32)
    n = q.shape[0]
    out = bytearray((n + 1) // 2)
    for i in range(0, n, 2):
        lo = int(q[i]) + 8
        hi = int(q[i + 1]) + 8 if i + 1 < n else 0
        out[i // 2] = (lo & 0x0F) | ((hi & 0x0F) << 4)
    return bytes(out), scale


def pack_with_scale(arr: np.ndarray, quant: int) -> bytes:
    """Per-tensor single scale (legacy int4 / int8)."""
    if quant == 4:
        payload, scale = quant_int4(arr)
    else:
        payload, scale = quant_int8(arr)
    return struct.pack("<f", scale) + payload


def pack_int4_grouped(arr: np.ndarray, group_size: int = 32) -> bytes:
    """Group-scaled int4 matching C rabbit_pack_int4_grouped layout."""
    f = arr.astype(np.float32).ravel()
    n = int(f.size)
    if group_size <= 0:
        group_size = 32
    n_groups = (n + group_size - 1) // group_size
    scales = np.zeros(n_groups, dtype=np.float32)
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, n)
        max_abs = float(np.max(np.abs(f[start:end]))) if end > start else 0.0
        scales[g] = max_abs / 7.0 + 1e-8

    q = np.empty(n, dtype=np.int32)
    for i in range(n):
        g = i // group_size
        q[i] = int(np.clip(np.round(f[i] / scales[g]), -8, 7))

    packed = bytearray((n + 1) // 2)
    for i in range(0, n, 2):
        lo = int(q[i]) + 8
        hi = int(q[i + 1]) + 8 if i + 1 < n else 0
        packed[i // 2] = (lo & 0x0F) | ((hi & 0x0F) << 4)

    return struct.pack("<I", group_size) + scales.tobytes() + bytes(packed)
