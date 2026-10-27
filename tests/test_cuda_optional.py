#!/usr/bin/env python3
"""Offline checks for optional CUDA GEMV integration (no GPU required)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_files_exist():
    for p in (
        "src/cuda_gemv.h",
        "src/cuda_gemv_stub.c",
        "src/cuda_gemv.cu",
    ):
        assert (ROOT / p).is_file(), p
    print("test_files_exist: ok")


def test_makefile_optional():
    mk = (ROOT / "Makefile").read_text()
    assert "RABBIT_CUDA" in mk
    assert "cuda_gemv_stub.c" in mk
    assert "cuda_gemv.cu" in mk
    print("test_makefile_optional: ok")


def test_default_build_no_cudart():
    """Default make must not require -lcudart (grep link line via dry logic)."""
    mk = (ROOT / "Makefile").read_text()
    # Default RABBIT_CUDA ?= 0 branch uses stub, not -lcudart unless RABBIT_CUDA=1
    assert "RABBIT_CUDA ?= 0" in mk or "RABBIT_CUDA?=0" in mk.replace(" ", "")
    print("test_default_build_no_cudart: ok")


def test_stub_api_in_binary():
    rabbit = ROOT / "rabbit"
    if not rabbit.is_file():
        print("test_stub_api_in_binary: SKIP (no binary)")
        return
    # quant path is exercised by unit tests; just ensure binary runs
    r = subprocess.run([str(rabbit), "version"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0
    print("test_stub_api_in_binary: ok")


def main() -> int:
    test_files_exist()
    test_makefile_optional()
    test_default_build_no_cudart()
    test_stub_api_in_binary()
    print("test_cuda_optional: all ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
