#!/usr/bin/env python3
"""Offline test for tools/convert_hf_stream.py against DeepSeek-MoE micro fixture."""
from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _header_fields(data: bytes):
    # <8sIIIIIIIIQQQII64s128s
    off = 8
    vals = struct.unpack_from("<IIIIIIII", data, off)
    # version, header_size, n_layers, hidden, vocab, n_experts, top_k, expert_dim
    n_layers, hidden, vocab, n_experts = vals[2], vals[3], vals[4], vals[5]
    off = 8 + 4 * 8
    dense_size, expert_size, total_experts = struct.unpack_from("<QQQ", data, off)
    off = 8 + 4 * 8 + 8 * 3
    dense_q, expert_q = struct.unpack_from("<II", data, off)
    hdr_size = vals[1]
    n_heads, n_kv, hd = struct.unpack_from("<III", data, hdr_size - 128)
    n_shared = struct.unpack_from("<I", data, hdr_size - 128 + 12)[0]
    return {
        "n_layers": n_layers,
        "hidden": hidden,
        "vocab": vocab,
        "n_experts": n_experts,
        "dense_size": dense_size,
        "expert_size": expert_size,
        "total_experts": total_experts,
        "dense_q": dense_q,
        "expert_q": expert_q,
        "n_heads": n_heads,
        "n_kv": n_kv,
        "hd": hd,
        "n_shared": n_shared,
        "hdr_size": hdr_size,
    }


def _convert_and_check(quant: int) -> Path:
    out = ROOT / f"models/deepseek-moe-stream-q{quant}.rabbit"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/convert_hf_stream.py"),
            "--model-dir",
            str(ROOT / "models/hf_deepseek_moe_micro"),
            "--out",
            str(out),
            "--quant",
            str(quant),
            "--name",
            "deepseek-moe-micro",
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
    assert "shared=1" in combined or "shared=True" in combined or "+1 shared" in combined, combined

    data = out.read_bytes()
    assert data[:8] == b"RABBIT01", "bad magic"
    h = _header_fields(data)
    assert h["n_experts"] == 4, h
    assert h["n_shared"] == 1, h
    assert h["n_layers"] == 2, h
    assert (h["n_heads"], h["n_kv"], h["hd"]) == (4, 2, 16), h
    assert h["dense_q"] == 0, h
    assert h["expert_q"] == quant, h
    # L * (E + S) records
    assert h["total_experts"] == h["expert_size"] * 2 * (4 + 1), h

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
        print("test_stream_deepseek: SKIP (pip install safetensors)")
        return 0

    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "tools/create_deepseek_moe_micro.py"),
            "--out-dir",
            str(ROOT / "models/hf_deepseek_moe_micro"),
            "--seed",
            "13",
        ],
        cwd=ROOT,
    )
    st = ROOT / "models/hf_deepseek_moe_micro/model.safetensors"
    if not st.is_file():
        print("test_stream_deepseek: SKIP (no model.safetensors written)")
        return 0

    for q in (4, 0):
        path = _convert_and_check(q)
        print(f"  quant={q} -> {path.name} ok")

    print("test_stream_deepseek: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
