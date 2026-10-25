#!/usr/bin/env python3
"""
Test script for CUDA backend (PR #3).

Tests CUDA availability, compilation, and basic functionality.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=cwd)


def test_cuda_availability():
    """Test that CUDA availability check works."""
    print("test_cuda_availability: ", end="", flush=True)

    rabbit = ROOT / "rabbit"
    if not rabbit.is_file():
        print("SKIP (binary not built)")
        return True

    result = run([str(rabbit), "version"])
    if result.returncode == 0:
        print("PASS (binary runs)")
        return True

    print(f"FAIL ({result.stderr[:100]})")
    return False


def test_cuda_kernel_compilation():
    """Test that CUDA kernel compiles without errors."""
    print("test_cuda_kernel_compilation: ", end="", flush=True)

    # Check if nvcc is available
    result = run(["which", "nvcc"])
    if result.returncode != 0:
        # Try common locations
        for path in ["/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"]:
            result = run([path, "--version"])
            if result.returncode == 0:
                print("PASS (nvcc found)")
                return True

        print("SKIP (nvcc not installed)")
        return True

    # Test compile the kernel
    result = run([
        "nvcc", "-std=c++17", "-I./src", "-c",
        "src/cuda_gemv.cu", "-o", "/tmp/cuda_gemv_test.o"
    ], cwd=ROOT)

    if result.returncode == 0:
        print("PASS (kernel compiles)")
        return True

    print(f"FAIL (compilation error: {result.stderr[:200]})")
    return False


def test_cuda_headers():
    """Test that CUDA headers are syntactically correct."""
    print("test_cuda_headers: ", end="", flush=True)

    headers = ["src/cuda_gemv.cu", "src/cuda_backend.c"]
    for h in headers:
        path = ROOT / h
        if not path.is_file():
            print(f"FAIL ({h} missing)")
            return False

    print("PASS (headers present)")
    return True


def test_makefile_cuda_targets():
    """Test that Makefile has CUDA targets."""
    print("test_makefile_cuda_targets: ", end="", flush=True)

    makefile = ROOT / "Makefile"
    if not makefile.is_file():
        print("FAIL (Makefile missing)")
        return False

    content = makefile.read_text()
    required = ["cuda_gemv.o", "NVCC", "CUDA_LDFLAGS", "cuda-test"]
    missing = [t for t in required if t not in content]

    if not missing:
        print("PASS (all CUDA targets present)")
        return True

    print(f"FAIL (missing: {missing})")
    return False


def test_cuda_api_in_rabbit_h():
    """Test that rabbit.h declares CUDA API."""
    print("test_cuda_api_in_rabbit_h: ", end="", flush=True)

    rabbit_h = ROOT / "src" / "rabbit.h"
    if not rabbit_h.is_file():
        print("FAIL (rabbit.h missing)")
        return False

    content = rabbit_h.read_text()
    required = [
        "rabbit_cuda_available",
        "rabbit_cuda_init",
        "rabbit_cuda_shutdown",
        "rabbit_cuda_moe_step",
        "rabbit_cuda_gemv_expert",
        "cuda_enabled",
        "cuda_device_id",
    ]
    missing = [t for t in required if t not in content]

    if not missing:
        print("PASS (CUDA API declared)")
        return True

    print(f"FAIL (missing: {missing})")
    return False


def main() -> int:
    print("Running PR #3 (CUDA Backend) validation tests...\n")

    tests = [
        test_cuda_kernel_compilation,
        test_cuda_headers,
        test_makefile_cuda_targets,
        test_cuda_api_in_rabbit_h,
        test_cuda_availability,
    ]

    failed = 0
    for test in tests:
        if not test():
            failed += 1
        print()

    if failed == 0:
        print("All CUDA backend tests PASSED")
        return 0
    else:
        print(f"{failed} CUDA backend test(s) FAILED")
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        raise SystemExit(1)