#!/usr/bin/env python3
"""Low-level quant/pack helpers for Rabbit conversion."""
from __future__ import annotations

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
    reserved = bytearray(128)
    # Optional GQA metadata in reserved[0:12]
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


def build_demo_model(out_path: Path, hidden_dim: int = 128, n_experts: int = 16, quant: int = 4) -> None:
    n_layers = 6
    vocab_size = 256
    experts_per_token = 2
    expert_dim = hidden_dim * 2

    dense = np.random.randn(hidden_dim * 256).astype(np.float32)
    dense_blob = pack_with_scale(dense, 8)

    expert_blobs = []
    for e in range(n_experts):
        mat = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        mat += e * 0.01
        if quant == 4:
            expert_blobs.append(pack_int4_grouped(mat))
        else:
            expert_blobs.append(pack_with_scale(mat, quant if quant else 8))

    expert_size = max(len(b) for b in expert_blobs)
    expert_blobs = [b + b"\0" * (expert_size - len(b)) for b in expert_blobs]

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
            dense_quant=8,
            expert_quant=quant if quant else 8,
            name="demo-moe",
        )
        f.write(dense_blob)
        for b in expert_blobs:
            f.write(b)

    print(f"[demo] {out_path}  quant={quant}  experts={n_experts}  hidden={hidden_dim}")


def load_hf_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        return json.load(f)


def collect_tensors(model_dir: Path) -> Dict[str, np.ndarray]:
    tensors: Dict[str, np.ndarray] = {}

    npz = model_dir / "model.npz"
    if npz.is_file():
        print(f"[load] {npz.name}")
        data = np.load(npz)
        for key in data.files:
            tensors[key] = data[key]
        return tensors

    try:
        from safetensors import safe_open
    except ImportError:
        print("pip install safetensors numpy  (or provide model.npz)", file=sys.stderr)
        sys.exit(1)

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors or model.npz in {model_dir}")
    for path in files:
        print(f"[load] {path.name}")
        with safe_open(path, framework="np") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def classify_tensors(tensors: Dict[str, np.ndarray]):
    dense = {}
    experts: Dict[int, Dict[str, np.ndarray]] = {}

    expert_markers = (
        ".experts.",
        "block_sparse_moe.experts",
        "mlp.experts",
        "moe.experts",
    )

    for name, arr in tensors.items():
        lower = name.replace("/", ".")
        is_expert = any(m in lower for m in expert_markers)

        if not is_expert:
            dense[name] = arr
            continue

        parts = lower.split(".")
        idx = None
        for i, p in enumerate(parts):
            if p in ("experts", "expert") and i + 1 < len(parts) and parts[i + 1].isdigit():
                idx = int(parts[i + 1])
                break

        if idx is None:
            dense[name] = arr
            continue

        experts.setdefault(idx, {})[name] = arr

    return dense, experts


def collect_experts_per_layer(
    tensors: Dict[str, np.ndarray],
    n_layers: int,
    n_experts: int,
) -> List[List[Dict[str, np.ndarray]]]:
    per_layer: List[List[Dict[str, np.ndarray]]] = [
        [{} for _ in range(n_experts)] for _ in range(n_layers)
    ]

    expert_markers = (
        ".experts.",
        "block_sparse_moe.experts",
        "mlp.experts",
        "moe.experts",
    )

    def _parse_layer_expert(name: str) -> Optional[Tuple[int, int]]:
        lower = name.replace("/", ".").lower()
        if not any(m in lower for m in expert_markers):
            return None
        parts = lower.split(".")
        layer_idx: Optional[int] = None
        expert_idx: Optional[int] = None
        for i, p in enumerate(parts):
            if p in ("layers", "layer", "h") and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer_idx = int(parts[i + 1])
            if p in ("experts", "expert") and i + 1 < len(parts) and parts[i + 1].isdigit():
                expert_idx = int(parts[i + 1])
        if expert_idx is None:
            return None
        return (layer_idx if layer_idx is not None else -1, expert_idx)

    shared: Dict[int, Dict[str, np.ndarray]] = {}
    saw_layered = False

    for name, arr in tensors.items():
        parsed = _parse_layer_expert(name)
        if parsed is None:
            continue
        layer_idx, expert_idx = parsed
        if expert_idx < 0 or expert_idx >= n_experts:
            continue
        if layer_idx >= 0:
            if layer_idx >= n_layers:
                continue
            per_layer[layer_idx][expert_idx][name] = arr
            saw_layered = True
        else:
            shared.setdefault(expert_idx, {})[name] = arr

    if not saw_layered and shared:
        for L in range(n_layers):
            for e in range(n_experts):
                if e in shared:
                    per_layer[L][e] = dict(shared[e])
        print(
            f"[v2] shared expert pool replicated across {n_layers} layers "
            f"({n_experts} experts → {n_layers * n_experts} records)"
        )
    else:
        filled = sum(1 for L in range(n_layers) for e in range(n_experts) if per_layer[L][e])
        print(f"[v2] per-layer experts: {filled}/{n_layers * n_experts} slots populated")

    return per_layer


def pack_dense(dense: Dict[str, np.ndarray], quant: int = 8) -> bytes:
    pieces = []
    for name in sorted(dense.keys()):
        pieces.append(pack_with_scale(dense[name], quant))
    return b"".join(pieces)


def _prefer_name(name: str) -> int:
    n = name.lower()
    if "down_proj" in n or "w2" in n:
        return 100
    if "up_proj" in n or "w3" in n or "w1" in n:
        return 50
    if "gate_proj" in n or "gate_up" in n:
        return 40
    return 10


def _order_key(name: str) -> int:
    n = name.lower()
    if "gate" in n and "proj" in n:
        return 0
    if "up" in n or "w1" in n or "w3" in n:
        return 1
    if "down" in n or "w2" in n:
        return 2
    return 3


def pack_one_expert(expert_tensors: Dict[str, np.ndarray], quant: int, layout: str,
                    hidden_dim: int) -> bytes:
    q = quant if quant else 8
    if layout == "square":
        candidates = []
        for name, arr in expert_tensors.items():
            a = np.asarray(arr)
            if a.ndim == 2:
                candidates.append((name, a))
        if candidates:
            candidates.sort(key=lambda x: (_prefer_name(x[0]), x[1].size), reverse=True)
            mat = candidates[0][1]
            flat = mat.astype(np.float32).ravel()
            need = hidden_dim * hidden_dim
            if flat.size < need:
                flat = np.pad(flat, (0, need - flat.size))
            else:
                flat = flat[:need]
            return pack_with_scale(flat.reshape(hidden_dim, hidden_dim), q)
        flat = np.concatenate([np.asarray(v).astype(np.float32).ravel() for v in expert_tensors.values()])
        need = hidden_dim * hidden_dim
        if flat.size < need:
            flat = np.pad(flat, (0, need - flat.size))
        return pack_with_scale(flat[:need].reshape(hidden_dim, hidden_dim), q)

    ordered = sorted(expert_tensors.items(), key=lambda kv: _order_key(kv[0]))
    pieces = [pack_with_scale(np.asarray(v), q) for _, v in ordered]
    return b"".join(pieces)


def _find(tensors: Dict[str, np.ndarray], *substrings: str) -> Optional[np.ndarray]:
    for name, arr in tensors.items():
        n = name.lower()
        if all(s in n for s in substrings):
            return np.asarray(arr, dtype=np.float32)
    return None


def _find_layer(tensors: Dict[str, np.ndarray], layer: int, *substrings: str) -> Optional[np.ndarray]:
    markers = (f"layers.{layer}.", f"layer.{layer}.", f".{layer}.")
    for name, arr in tensors.items():
        n = name.lower()
        if not any(m in n for m in markers):
            continue
        if all(s in n for s in substrings):
            return np.asarray(arr, dtype=np.float32)
    return None
