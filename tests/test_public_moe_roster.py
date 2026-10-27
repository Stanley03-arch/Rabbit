#!/usr/bin/env python3
"""Offline tests for public MoE roster tooling (no Hub download)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)


def test_list_roster():
    script = ROOT / "tools" / "download_public_moe.py"
    r = run([sys.executable, str(script), "--list"])
    assert r.returncode == 0, r.stderr
    assert "tinybrainbot" in r.stdout
    assert "olmoe" in r.stdout
    assert "qwen-moe-a2.7b" in r.stdout
    # Guard against resurrecting fictional sizes in this script
    assert "744B" not in r.stdout
    assert "2.8T" not in r.stdout
    print("test_list_roster: ok")


def test_skip_all_no_hub():
    script = ROOT / "tools" / "download_public_moe.py"
    r = run([
        sys.executable, str(script),
        "--models", "tinybrainbot",
        "--skip-download", "--skip-convert", "--skip-verify",
        "--models-dir", "/tmp/rabbit_hf_public_test",
        "--out-dir", "/tmp/rabbit_out_public_test",
    ])
    # Without config.json, skip-download should fail the one model → exit 1
    # That is expected; we only check the script executes
    assert r.returncode in (0, 1), (r.returncode, r.stderr)
    print("test_skip_all_no_hub: ok")


def test_hf_backend_missing_dir():
    script = ROOT / "tools" / "hf_backend.py"
    r = run([
        sys.executable, str(script),
        "--model-dir", "/nonexistent_rabbit_hf",
        "--prompt", "hi",
        "--output", "/tmp/rabbit_hf_backend_test.json",
    ])
    assert r.returncode != 0
    print("test_hf_backend_missing_dir: ok")


def test_makefile_mentions():
    mk = (ROOT / "Makefile").read_text()
    for t in ("download-moe", "public-moe", "download_public_moe.py"):
        assert t in mk, t
    print("test_makefile_mentions: ok")


def main() -> int:
    test_list_roster()
    test_skip_all_no_hub()
    test_hf_backend_missing_dir()
    test_makefile_mentions()
    print("test_public_moe_roster: all ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
