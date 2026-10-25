#!/usr/bin/env python3
"""M2: convert HF-like micro MoE with --layout v2 and exercise C ref-v2 path."""

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
        print("test_m2: missing ./rabbit — run make", file=sys.stderr)
        return 1

    hf_dir = ROOT / "models" / "hf_micro_moe"
    out = ROOT / "models" / "micro-hf-v2.rabbit"

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
            "8",
        ]
    )
    if r.returncode != 0:
        print(r.stdout, r.stderr, file=sys.stderr)
        return 1
    if not out.is_file():
        print("test_m2: missing output", file=sys.stderr)
        return 1

    # C probe-v2 must load and produce tokens
    r = run(
        [
            str(rabbit),
            "probe-v2",
            "--model",
            str(out),
            "--prompt",
            "hi",
            "--tokens",
            "16",
        ]
    )
    if r.returncode != 0:
        print("test_m2: probe-v2 failed", r.stdout, r.stderr, file=sys.stderr)
        return 1
    m = re.search(r"PROBE_V2.*ids=([0-9,]+)", r.stdout + r.stderr)
    if not m:
        print("test_m2: no PROBE_V2 line", r.stdout, file=sys.stderr)
        return 1
    c_ids = [int(x) for x in m.group(1).split(",") if x]

    # NumPy oracle on same file
    model = load_ref_v2(out)
    py_ids = greedy_generate(model, "hi", 16)
    if c_ids != py_ids:
        print("test_m2: C/NumPy mismatch", file=sys.stderr)
        print(" c", c_ids, file=sys.stderr)
        print("py", py_ids, file=sys.stderr)
        return 1

    print("test_m2: ok (HF micro → layout v2 → C matches NumPy)")
    print(f"  tokens={c_ids[:12]}...")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("test_m2: FAIL", e, file=sys.stderr)
        raise SystemExit(1)
