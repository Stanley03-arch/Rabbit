#!/usr/bin/env python3
"""Convert Mixtral-named micro fixture and verify GQA header + probe."""
from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    subprocess.check_call(
        [sys.executable, str(ROOT / "tools/create_mixtral_micro.py"),
         "--out-dir", str(ROOT / "models/hf_mixtral_micro"), "--seed", "7"],
        cwd=ROOT,
    )
    out = ROOT / "models/mixtral-micro.rabbit"
    subprocess.check_call(
        [sys.executable, str(ROOT / "tools/convert_hf_to_rabbit.py"),
         "--model-dir", str(ROOT / "models/hf_mixtral_micro"),
         "--out", str(out), "--layout", "v2", "--quant", "4"],
        cwd=ROOT,
    )
    data = out.read_bytes()
    assert data[:8] == b"RABBIT01"
    hdr_size = struct.unpack_from("<I", data, 12)[0]
    n_heads, n_kv, hd = struct.unpack_from("<III", data, hdr_size - 128)
    assert (n_heads, n_kv, hd) == (4, 2, 16), (n_heads, n_kv, hd)

    rabbit = ROOT / "rabbit"
    if rabbit.is_file():
        r = subprocess.run(
            [str(rabbit), "probe-v2", "--model", str(out), "--prompt", "hi", "--tokens", "8"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            return r.returncode
        assert "PROBE_V2" in r.stdout or "PROBE_V2" in r.stderr

    print("test_mixtral_micro: ok")
    print(f"  GQA heads={n_heads} kv={n_kv} hd={hd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
