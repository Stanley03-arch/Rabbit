#!/usr/bin/env python3
"""Unit tests for tools/rabbit_name_maps.py (no safetensors required)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from rabbit_name_maps import (  # noqa: E402
    detect_expert_scheme,
    detect_shared_experts,
    resolve_expert_triple_names,
    resolve_router_name,
    resolve_shared_expert_names,
)


def test_mixtral_scheme() -> None:
    names = {
        "model.layers.0.block_sparse_moe.experts.0.w1.weight",
        "model.layers.0.block_sparse_moe.experts.0.w2.weight",
        "model.layers.0.block_sparse_moe.experts.0.w3.weight",
        "model.layers.0.block_sparse_moe.gate.weight",
    }
    assert detect_expert_scheme(lambda n: n in names) == "mixtral-block_sparse_moe"


def test_qwen_scheme() -> None:
    names = {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.gate.weight",
    }
    assert detect_expert_scheme(lambda n: n in names) == "qwen-mlp.experts"


def test_shared_experts() -> None:
    names = {
        "model.layers.0.mlp.shared_experts.gate_proj.weight",
        "model.layers.0.mlp.shared_experts.up_proj.weight",
        "model.layers.0.mlp.shared_experts.down_proj.weight",
    }
    assert detect_shared_experts(lambda n: n in names) is True
    assert detect_shared_experts(lambda n: False) is False


def test_resolve_nonempty() -> None:
    assert len(resolve_expert_triple_names(0, 0)) >= 5
    assert len(resolve_router_name(1)) >= 4
    assert len(resolve_shared_expert_names(2)) >= 3


def main() -> int:
    test_mixtral_scheme()
    test_qwen_scheme()
    test_shared_experts()
    test_resolve_nonempty()
    print("test_name_maps: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
