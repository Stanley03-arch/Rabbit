#!/usr/bin/env python3
"""Unit test: OLMoE exact tensor-map classifier (no model download)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from convert_hf_to_rabbit import (  # noqa: E402
    classify_tensors_olmoe,
    pack_one_expert_olmoe,
)


def test_classify_olmoe():
    H = 8
    tensors = {
        "model.embed_tokens.weight": np.zeros((32, H), dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.zeros((H, H), dtype=np.float32),
        "model.layers.0.mlp.gate.weight": np.zeros((4, H), dtype=np.float32),
        "model.layers.0.mlp.experts.0.gate_proj.weight": np.ones((16, H), dtype=np.float32),
        "model.layers.0.mlp.experts.0.up_proj.weight": np.ones((16, H), dtype=np.float32) * 2,
        "model.layers.0.mlp.experts.0.down_proj.weight": np.ones((H, 16), dtype=np.float32) * 3,
        "model.layers.0.mlp.experts.1.gate_proj.weight": np.ones((16, H), dtype=np.float32) * 4,
        "model.layers.1.mlp.experts.0.gate_proj.weight": np.ones((16, H), dtype=np.float32) * 9,  # same E=0
        "model.norm.weight": np.ones(H, dtype=np.float32),
        "lm_head.weight": np.zeros((32, H), dtype=np.float32),
    }
    dense, experts = classify_tensors_olmoe(tensors)

    assert "model.embed_tokens.weight" in dense
    assert "model.layers.0.mlp.gate.weight" in dense  # router stays dense
    assert "lm_head.weight" in dense
    assert 0 in experts and 1 in experts
    assert "gate_proj.weight" in experts[0]
    assert "up_proj.weight" in experts[0]
    assert "down_proj.weight" in experts[0]
    # first-seen wins for expert 0 (layer 0, not layer 1)
    assert float(experts[0]["gate_proj.weight"].max()) == 1.0
    assert len(experts) == 2


def test_pack_olmoe_order():
    H = 4
    expert = {
        "gate_proj.weight": np.full((2, H), 1.0, dtype=np.float32),
        "up_proj.weight": np.full((2, H), 2.0, dtype=np.float32),
        "down_proj.weight": np.full((H, 2), 3.0, dtype=np.float32),
    }
    blob = pack_one_expert_olmoe(expert, quant=8, hidden_dim=H)
    assert len(blob) >= 4 + H * H
    # scale is first float
    import struct

    scale = struct.unpack_from("<f", blob, 0)[0]
    assert scale > 0


def main() -> int:
    test_classify_olmoe()
    test_pack_olmoe_order()
    print("test_olmoe_map: ok")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("test_olmoe_map: FAIL", e, file=sys.stderr)
        raise SystemExit(1)
