#!/usr/bin/env python3
"""Exact parity: NumPy ref_forward vs `rabbit probe` on ref-v1."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ref_forward import forward, load_ref  # noqa: E402


def parse_probe(text: str) -> dict:
    m = re.search(
        r"PROBE prompt=(\S+) dim=(\d+) sum=([-+0-9.eE]+) l1=([-+0-9.eE]+) l2=([-+0-9.eE]+) first8=([-0-9.,eE]+)",
        text,
    )
    if not m:
        raise RuntimeError("probe line not found in:\n" + text[-2000:])
    first8 = [float(x) for x in m.group(6).split(",")]
    return {
        "prompt": m.group(1),
        "dim": int(m.group(2)),
        "sum": float(m.group(3)),
        "l1": float(m.group(4)),
        "l2": float(m.group(5)),
        "first8": first8,
    }


def run_probe(model: Path, prompt: str) -> dict:
    rabbit = ROOT / "rabbit"
    proc = subprocess.run(
        [str(rabbit), "probe", "--model", str(model), "--prompt", prompt],
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"probe failed rc={proc.returncode}: {out[-1500:]}")
    return parse_probe(out)


def close(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main() -> int:
    model_path = ROOT / "models" / "ref-v1.rabbit"
    if not model_path.is_file():
        subprocess.check_call(
            [sys.executable, str(ROOT / "tools" / "create_ref_model.py"), "--out", str(model_path)]
        )

    prompt = "hi"
    m = load_ref(model_path)
    ref = forward(m, prompt)
    c = run_probe(model_path, prompt)

    assert c["dim"] == m["hidden_dim"]
    assert close(c["sum"], ref["sum"]), (c["sum"], ref["sum"])
    assert close(c["l1"], ref["l1"]), (c["l1"], ref["l1"])
    assert close(c["l2"], ref["l2"]), (c["l2"], ref["l2"])
    for i, (cv, rv) in enumerate(zip(c["first8"], ref["first8"])):
        assert close(cv, rv, 1e-4), (i, cv, rv)

    # second prompt
    ref2 = forward(m, "ab")
    c2 = run_probe(model_path, "ab")
    assert close(c2["l2"], ref2["l2"]), (c2["l2"], ref2["l2"])

    print("parity-ref: ok (exact C vs NumPy within 1e-4)")
    print(f"  prompt=hi  l2_ref={ref['l2']:.6f}  l2_c={c['l2']:.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("parity-ref: FAIL", e, file=sys.stderr)
        raise SystemExit(1)
