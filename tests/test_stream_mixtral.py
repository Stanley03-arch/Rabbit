#!/usr/bin/env python3
"""Offline test for tools/convert_hf_stream.py against Mixtral micro fixture."""
from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _convert_and_check(quant: int) -> Path:
    out = ROOT / f"models/mixtral-stream-q{quant}.rabbit"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "tools/convert_hf_stream.py"),
            "--model-dir",
            str(ROOT / "models/hf_mixtral_micro"),
            "--out",
            str(out),
            "--quant",
            str(quant),
        ],
        cwd=ROOT,
    )
    data = out.read_bytes()
    assert data[:8] == b"RABBIT01", "bad magic"
    hdr_size = struct.unpack_from("<I", data, 12)[0]
    n_heads, n_kv, hd = struct.unpack_from("<III", data, hdr_size - 128)
    assert (n_heads, n_kv, hd) == (4, 2, 16), (n_heads, n_kv, hd)

    expert_quant = struct.unpack_from("<I", data, 8 + 4 * 8 + 8 * 3 + 4)[0]
    # layout: magic(8) + 8xI + 3xQ + dense_quant(I) + expert_quant(I)
    # safer: read from known offset after total_experts_size
    # header pack: <8sIIIIIIIIQQQII64s128s
    # expert_quant is the second of the two I after three Q
    off = 8 + 4 * 8 + 8 * 3  # after magic + 8 ints + 3 quads
    dense_q, expert_q = struct.unpack_from("<II", data, off)
    assert dense_q == 0, dense_q
    assert expert_q == quant, (expert_q, quant)

    rabbit = ROOT / "rabbit"
    if rabbit.is_file():
        r = subprocess.run(
            [
                str(rabbit),
                "probe-v2",
                "--model",
                str(out),
                "--prompt",
                "hi",
                "--tokens",
                "8",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            raise SystemExit(r.returncode)
        assert "PROBE_V2" in r.stdout or "PROBE_V2" in r.stderr

    return out


def main() -> int:
    try:
        import safetensors  # noqa: F401
    except ImportError:
        print("test_stream_mixtral: SKIP (pip install safetensors)")
        return 0

    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "tools/create_mixtral_micro.py"),
            "--out-dir",
            str(ROOT / "models/hf_mixtral_micro"),
            "--seed",
            "7",
        ],
        cwd=ROOT,
    )
    st = ROOT / "models/hf_mixtral_micro/model.safetensors"
    if not st.is_file():
        print("test_stream_mixtral: SKIP (no model.safetensors written)")
        return 0

    for q in (4, 0):
        path = _convert_and_check(q)
        print(f"  quant={q} -> {path.name} ok")

    print("test_stream_mixtral: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
