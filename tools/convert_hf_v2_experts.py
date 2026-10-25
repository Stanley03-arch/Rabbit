#!/usr/bin/env python3
"""Helpers shared conceptually with convert_hf — per-layer expert collection."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def collect_experts_per_layer(
    tensors: Dict[str, np.ndarray], n_layers: int, n_experts: int
) -> List[List[Dict[str, np.ndarray]]]:
    """Return experts[layer][local_id] = {name: array}."""
    out: List[List[Dict[str, np.ndarray]]] = [
        [{} for _ in range(n_experts)] for _ in range(n_layers)
    ]

    for name, arr in tensors.items():
        lower = name.replace("/", ".")
        if ".experts." not in lower and "block_sparse_moe.experts" not in lower:
            continue
        parts = lower.split(".")
        layer = None
        eid = None
        for i, p in enumerate(parts):
            if p in ("layers", "layer") and i + 1 < len(parts) and parts[i + 1].isdigit():
                layer = int(parts[i + 1])
            if p in ("experts", "expert") and i + 1 < len(parts) and parts[i + 1].isdigit():
                eid = int(parts[i + 1])
        if layer is None or eid is None:
            continue
        if layer < 0 or layer >= n_layers or eid < 0 or eid >= n_experts:
            continue
        out[layer][eid][name] = arr

    return out
