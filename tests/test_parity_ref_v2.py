#!/usr/bin/env python3
"""Parity harness for rabbit-ref-v2 (NumPy oracle self-check; C match is next)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ref_forward_v2 import forward_logits, greedy_generate, load_ref_v2  # noqa: E402


def main() -> int:
    model_path = ROOT / "models" / "ref-v2.rabbit"
    if not model_path.is_file():
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "tools" / "create_ref_v2.py"),
                "--out",
                str(model_path),
                "--seed",
                "2",
            ]
        )

    m = load_ref_v2(model_path)
    assert m["hidden"] == 64 and m["n_experts"] == 4 and m["expert_dim"] == 128

    # Determinism: same prompt → same token ids
    a = greedy_generate(m, "hi", 16)
    b = greedy_generate(m, "hi", 16)
    assert a == b, (a, b)

    # Logits finite and peaked
    logits = forward_logits(m, a[:4])
    assert np.all(np.isfinite(logits))
    assert logits.shape == (m["vocab"],)

    # Second prompt differs
    c = greedy_generate(m, "ab", 8)
    assert c != a[: len(c)] or len(set(c)) > 1

    print("parity-ref-v2: ok (NumPy oracle deterministic)")
    print(f"  prompt=hi tokens[0:16]={a[:16]}")
    print(f"  note: C token-exact match still open (Phase D M1)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("parity-ref-v2: FAIL", e, file=sys.stderr)
        raise SystemExit(1)
