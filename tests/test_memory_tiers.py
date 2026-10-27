#!/usr/bin/env python3
"""Sanity checks for multitier memory docs/API surface."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_headers():
    h = (ROOT / "src" / "rabbit.h").read_text()
    assert "rabbit_expert_prefetch" in h
    assert "rabbit_cache_pin" in h
    assert "RABBIT_MAX_CACHE_SLOTS" in h
    assert "cache_hits" in h
    print("test_headers: ok")


def test_memory_c():
    t = (ROOT / "src" / "memory.c").read_text()
    assert "MADV_RANDOM" in t
    assert "MADV_WILLNEED" in t
    assert "rabbit_expert_prefetch" in t
    print("test_memory_c: ok")


def test_docs():
    assert (ROOT / "docs" / "MEMORY.md").is_file()
    print("test_docs: ok")


def main() -> int:
    test_headers()
    test_memory_c()
    test_docs()
    print("test_memory_tiers: all ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
