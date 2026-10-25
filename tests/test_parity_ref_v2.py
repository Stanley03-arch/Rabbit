#!/usr/bin/env python3
"""C vs NumPy token parity for rabbit-ref-v2 (Phase D M1)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ref_forward_v2 import greedy_generate, load_ref_v2  # noqa: E402


def ensure_model() -> Path:
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
    return model_path


def run_c_probe(rabbit: Path, model: Path, prompt: str, n_new: int) -> list[int]:
    proc = subprocess.run(
        [
            str(rabbit),
            "probe-v2",
            "--model",
            str(model),
            "--prompt",
            prompt,
            "--tokens",
            str(n_new),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"probe-v2 failed rc={proc.returncode}\n{out}")
    m = re.search(r"PROBE_V2\s+prompt=\S+\s+n=(\d+)\s+ids=([0-9,]+)", out)
    if not m:
        raise RuntimeError(f"could not parse PROBE_V2 line:\n{out}")
    ids = [int(x) for x in m.group(2).split(",") if x != ""]
    return ids


def main() -> int:
    model_path = ensure_model()
    rabbit = ROOT / "rabbit"
    if not rabbit.is_file():
        print("parity-ref-v2: rabbit binary missing — run `make`", file=sys.stderr)
        return 1

    m = load_ref_v2(model_path)
    prompts = ["hi", "ab", "MoE"]
    n_new = 32

    for prompt in prompts:
        py_ids = greedy_generate(m, prompt, n_new)
        c_ids = run_c_probe(rabbit, model_path, prompt, n_new)
        if py_ids != c_ids:
            print(f"parity-ref-v2: FAIL prompt={prompt!r}", file=sys.stderr)
            print(f"  numpy: {py_ids}", file=sys.stderr)
            print(f"  c:     {c_ids}", file=sys.stderr)
            # show first mismatch
            for i, (a, b) in enumerate(zip(py_ids, c_ids)):
                if a != b:
                    print(f"  first mismatch at index {i}: np={a} c={b}", file=sys.stderr)
                    break
            if len(py_ids) != len(c_ids):
                print(f"  len np={len(py_ids)} c={len(c_ids)}", file=sys.stderr)
            return 1

    print(f"parity-ref-v2: ok (C matches NumPy for {len(prompts)} prompts × {n_new} new tokens)")
    print(f"  sample prompt=hi ids[:16]={greedy_generate(m, 'hi', 16)[:16]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("parity-ref-v2: FAIL", e, file=sys.stderr)
        raise SystemExit(1)
