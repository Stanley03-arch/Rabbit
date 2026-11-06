#!/usr/bin/env python3
"""Offline test for tools/convert_hf_stream.py against Qwen-MoE micro fixture."""
from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _convert_and_check(quant: int) -> Path:
    out = ROOT / f"models/qwen-moe-stream-q{quant}.rabbit"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/convert_hf_stream.py"),
            "--model-dir",
            str(ROOT / "models/hf_qwen_moe_micro"),
            "--out",
            str(out),
            "--quant",
            str(quant),
            "--name",
            "qwen-moe-micro",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(proc.returncode)

    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "qwen" in combined.lower() or "mlp.experts" in combined.lower(), combined

    data = out.read_bytes()
    assert data[:8] == b"RABBIT01", "bad magic"
    hdr_size = struct.unpack_from("<I", data, 12)[0]
    n_heads, n_kv, hd = struct.unpack_from("<III", data, hdr_size - 128)
    assert (n_heads, n_kv, hd) == (4, 2, 16), (n_heads, n_kv, hd)

    off = 8 + 4 * 8 + 8 * 3
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
        print("test_stream_qwen_moe: SKIP (pip install safetensors)")
        return 0

    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "tools/create_qwen_moe_micro.py"),
            "--out-dir",
            str(ROOT / "models/hf_qwen_moe_micro"),
            "--seed",
            "11",
        ],
        cwd=ROOT,
    )
    st = ROOT / "models/hf_qwen_moe_micro/model.safetensors"
    if not st.is_file():
        print("test_stream_qwen_moe: SKIP (no model.safetensors written)")
        return 0

    for q in (4, 0):
        path = _convert_and_check(q)
        print(f"  quant={q} -> {path.name} ok")

    print("test_stream_qwen_moe: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
