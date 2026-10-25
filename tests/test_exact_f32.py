#!/usr/bin/env python3
"""Exact path: HF micro → layout v2 --quant 0 → C matches NumPy."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ref_forward_v2 import greedy_generate, load_ref_v2  # noqa: E402


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=180)


def main() -> int:
    rabbit = ROOT / "rabbit"
    if not rabbit.is_file():
        print("test_exact_f32: missing ./rabbit", file=sys.stderr)
        return 1

    hf_dir = ROOT / "models" / "hf_micro_moe"
    out = ROOT / "models" / "micro-hf-f32.rabbit"

    r = run([sys.executable, str(ROOT / "tools" / "create_hf_micro_moe.py"), "--out-dir", str(hf_dir), "--seed", "3"])
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return 1

    r = run(
        [
            sys.executable,
            str(ROOT / "tools" / "convert_hf_to_rabbit.py"),
            "--model-dir",
            str(hf_dir),
            "--out",
            str(out),
            "--layout",
            "v2",
            "--quant",
            "0",
        ]
    )
    if r.returncode != 0:
        print(r.stdout, r.stderr, file=sys.stderr)
        return 1

    r = run([str(rabbit), "probe-v2", "--model", str(out), "--prompt", "hi", "--tokens", "24"])
    if r.returncode != 0:
        print("probe failed", r.stdout, r.stderr, file=sys.stderr)
        return 1
    m = re.search(r"PROBE_V2.*ids=([0-9,]+)", r.stdout + r.stderr)
    if not m:
        print("no PROBE_V2", r.stdout, file=sys.stderr)
        return 1
    c_ids = [int(x) for x in m.group(1).split(",") if x]

    model = load_ref_v2(out)
    if model.get("expert_quant", 8) != 0:
        print("expected expert_quant=0", model.get("expert_quant"), file=sys.stderr)
        return 1
    py_ids = greedy_generate(model, "hi", 24)
    if c_ids != py_ids:
        print("C/NumPy mismatch", file=sys.stderr)
        print(" c", c_ids, file=sys.stderr)
        print("py", py_ids, file=sys.stderr)
        return 1

    print("test_exact_f32: ok (quant=0 experts, C matches NumPy)")
    print(f"  tokens={c_ids[:12]}...")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("test_exact_f32: FAIL", e, file=sys.stderr)
        raise SystemExit(1)
