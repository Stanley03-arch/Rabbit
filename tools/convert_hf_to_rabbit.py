#!/usr/bin/env python3
"""
convert_hf_to_rabbit.py

Convert open MoE models into the Rabbit binary format.

- int4 packing (default for experts when --quant 4)
- Explicit expert layout: [float32 scale][packed weights]
- --arch auto|olmoe  (olmoe uses the frozen tensor map in docs/tensor-map.md)
- --layout square|flat
- Demo mode (no download)

Usage:
  python tools/convert_hf_to_rabbit.py --demo --out models/demo-moe.rabbit
  python tools/convert_hf_to_rabbit.py --model-dir /path/to/hf --out models/m.rabbit --quant 4
  python tools/convert_hf_to_rabbit.py --model-dir /path/to/OLMoE --out models/olmoe.rabbit --arch olmoe
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

RABBIT_MAGIC = b"RABBIT01"
HEADER_SIZE = struct.calcsize(
    "8s" + "I" * 8 + "Q" * 3 + "I" * 2 + "64s" + "128s"
)


def write_header(f, **kw) -> None:
    name_bytes = kw["name"].encode("utf-8")[:63].ljust(64, b"\0")
    reserved = b"\0" * 128
    total_experts_size = kw["n_experts"] * kw["expert_size"]
    header = struct.pack(
        "<8sIIIIIIIIQQQII64s128s",
        RABBIT_MAGIC,
        1,
        HEADER_SIZE,
        kw["n_layers"],
        kw["hidden_dim"],
        kw["vocab_size"],
        kw["n_experts"],
        kw["experts_per_token"],
        kw["expert_dim"],
        kw["dense_size"],
        kw["expert_size"],
        total_experts_size,
        kw["dense_quant"],
        kw["expert_quant"],
        name_bytes,
        reserved,
    )
    f.write(header)


# ---------- quantization ----------

def quant_int8(arr: np.ndarray) -> Tuple[bytes, float]:
    f = arr.astype(np.float32).ravel()
    scale = float(np.max(np.abs(f)) / 127.0 + 1e-8)
    q = np.clip(np.round(f / scale), -128, 127).astype(np.int8)
    return q.tobytes(), scale


def quant_int4(arr: np.ndarray) -> Tuple[bytes, float]:
    """Pack two signed 4-bit values per byte (low nibble first)."""
    f = arr.astype(np.float32).ravel()
    scale = float(np.max(np.abs(f)) / 7.0 + 1e-8)
    q = np.clip(np.round(f / scale), -8, 7).astype(np.int32)
    n = q.shape[0]
    out = bytearray((n + 1) // 2)
    for i in range(0, n, 2):
        lo = int(q[i]) + 8
        hi = int(q[i + 1]) + 8 if i + 1 < n else 0
        out[i // 2] = (lo & 0x0F) | ((hi & 0x0F) << 4)
    return bytes(out), scale


def pack_with_scale(arr: np.ndarray, quant: int) -> bytes:
    if quant == 4:
        payload, scale = quant_int4(arr)
    else:
        payload, scale = quant_int8(arr)
    return struct.pack("<f", scale) + payload


# ---------- demo ----------

def build_demo_model(out_path: Path, hidden_dim: int = 128, n_experts: int = 16, quant: int = 4) -> None:
    n_layers = 6
    vocab_size = 256
    experts_per_token = 2
    expert_dim = hidden_dim * 2

    dense = np.random.randn(hidden_dim * 256).astype(np.float32)
    dense_blob = pack_with_scale(dense, 8)

    expert_blobs = []
    for e in range(n_experts):
        mat = np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        mat += e * 0.01
        expert_blobs.append(pack_with_scale(mat, quant))

    expert_size = max(len(b) for b in expert_blobs)
    expert_blobs = [b + b"\0" * (expert_size - len(b)) for b in expert_blobs]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        write_header(
            f,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            n_experts=n_experts,
            experts_per_token=experts_per_token,
            expert_dim=expert_dim,
            dense_size=len(dense_blob),
            expert_size=expert_size,
            dense_quant=8,
            expert_quant=quant,
            name="demo-moe",
        )
        f.write(dense_blob)
        for b in expert_blobs:
            f.write(b)

    print(f"[demo] {out_path}  quant={quant}  experts={n_experts}  hidden={hidden_dim}")
    print(f"       file={out_path.stat().st_size/1024:.1f} KB")


# ---------- HF conversion ----------

def load_hf_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        return json.load(f)


def find_safetensors(model_dir: Path) -> List[Path]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors in {model_dir}")
    return files


def collect_tensors(model_dir: Path) -> Dict[str, np.ndarray]:
    try:
        from safetensors import safe_open
    except ImportError:
        print("pip install safetensors numpy", file=sys.stderr)
        sys.exit(1)

    tensors = {}
    for path in find_safetensors(model_dir):
        print(f"[load] {path.name}")
        with safe_open(path, framework="np") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def classify_tensors_auto(tensors: Dict[str, np.ndarray]):
    """Heuristic detection for DeepSeek / Qwen / Mixtral / similar."""
    dense = {}
    experts: Dict[int, Dict[str, np.ndarray]] = {}

    expert_markers = (
        ".experts.",
        "block_sparse_moe.experts",
        "mlp.experts",
        "moe.experts",
        "shared_experts",
    )

    for name, arr in tensors.items():
        lower = name.replace("/", ".")
        is_expert = any(m in lower for m in expert_markers)

        if not is_expert:
            dense[name] = arr
            continue

        parts = lower.split(".")
        idx = None
        for i, p in enumerate(parts):
            if p in ("experts", "expert") and i + 1 < len(parts) and parts[i + 1].isdigit():
                idx = int(parts[i + 1])
                break
            if p.isdigit() and i > 0 and parts[i - 1] in ("experts", "expert"):
                idx = int(p)
                break

        if idx is None:
            dense[name] = arr
            continue

        experts.setdefault(idx, {})[name] = arr

    return dense, experts


# Exact OLMoE patterns — see docs/tensor-map.md
_OLMOE_EXPERT_RE = re.compile(
    r"(?:^|\.)model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
_OLMOE_EXPERT_ALT = re.compile(
    r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def classify_tensors_olmoe(tensors: Dict[str, np.ndarray]):
    """
    Exact tensor map for OLMoE-family checkpoints.

    Experts: model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight
    Everything else → dense (sorted-name concat).
    Expert id is global {E}; first-seen tensors win if layers collide.
    """
    dense = {}
    experts: Dict[int, Dict[str, np.ndarray]] = {}

    for name, arr in tensors.items():
        norm = name.replace("/", ".")
        m = _OLMOE_EXPERT_RE.search(norm) or _OLMOE_EXPERT_ALT.search(norm)
        if m:
            _layer, eid_s, proj = m.group(1), m.group(2), m.group(3)
            eid = int(eid_s)
            # Canonical key so packing order is stable
            key = f"{proj}.weight"
            slot = experts.setdefault(eid, {})
            if key not in slot:
                slot[key] = arr
            continue
        dense[name] = arr

    return dense, experts


def pack_dense(dense: Dict[str, np.ndarray], quant: int = 8) -> bytes:
    pieces = []
    for name in sorted(dense.keys()):
        pieces.append(pack_with_scale(dense[name], quant))
    return b"".join(pieces)


def pack_one_expert_olmoe(expert_tensors: Dict[str, np.ndarray], quant: int,
                          hidden_dim: int) -> bytes:
    """
    Phase A OLMoE packing (documented in docs/tensor-map.md):
      gate_proj → up_proj → down_proj, concat, pad/truncate to HxH, quantize.
    """
    order = ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
    parts = []
    for key in order:
        if key in expert_tensors:
            parts.append(np.asarray(expert_tensors[key], dtype=np.float32).ravel())
        else:
            # missing proj → skip (still deterministic)
            pass
    if not parts:
        # fallback: any tensors present
        for _, v in sorted(expert_tensors.items()):
            parts.append(np.asarray(v, dtype=np.float32).ravel())
    if not parts:
        flat = np.zeros(hidden_dim * hidden_dim, dtype=np.float32)
    else:
        flat = np.concatenate(parts)
    need = hidden_dim * hidden_dim
    if flat.size < need:
        flat = np.pad(flat, (0, need - flat.size))
    else:
        flat = flat[:need]
    mat = flat.reshape(hidden_dim, hidden_dim)
    return pack_with_scale(mat, quant)


def pack_one_expert(expert_tensors: Dict[str, np.ndarray], quant: int, layout: str,
                    hidden_dim: int, arch: str) -> bytes:
    if arch == "olmoe" and layout == "square":
        return pack_one_expert_olmoe(expert_tensors, quant, hidden_dim)

    if layout == "square":
        candidates = []
        for name, arr in expert_tensors.items():
            a = np.asarray(arr)
            if a.ndim == 2:
                candidates.append(a)
        if candidates:
            mat = max(candidates, key=lambda x: x.size)
            flat = mat.astype(np.float32).ravel()
            need = hidden_dim * hidden_dim
            if flat.size < need:
                flat = np.pad(flat, (0, need - flat.size))
            else:
                flat = flat[:need]
            mat = flat.reshape(hidden_dim, hidden_dim)
            return pack_with_scale(mat, quant)

        flat = np.concatenate(
            [np.asarray(v).astype(np.float32).ravel() for v in expert_tensors.values()]
        )
        need = hidden_dim * hidden_dim
        if flat.size < need:
            flat = np.pad(flat, (0, need - flat.size))
        mat = flat[:need].reshape(hidden_dim, hidden_dim)
        return pack_with_scale(mat, quant)

    pieces = [pack_with_scale(np.asarray(v), quant) for _, v in sorted(expert_tensors.items())]
    return b"".join(pieces)


def convert_hf(model_dir: Path, out_path: Path, max_experts: Optional[int],
               quant: int, layout: str, arch: str) -> None:
    config = load_hf_config(model_dir)
    print("[config]", {k: config.get(k) for k in (
        "model_type", "hidden_size", "num_hidden_layers", "num_local_experts",
        "num_experts_per_tok", "vocab_size", "moe_intermediate_size",
        "n_routed_experts", "num_experts", "intermediate_size",
    ) if k in config})
    print(f"[arch] {arch}")

    tensors = collect_tensors(model_dir)
    print(f"[info] {len(tensors)} tensors")

    if arch == "olmoe":
        dense, experts = classify_tensors_olmoe(tensors)
    else:
        dense, experts = classify_tensors_auto(tensors)
    print(f"[info] dense={len(dense)}  expert_groups={len(experts)}")

    hidden_dim = int(config.get("hidden_size") or config.get("n_embd") or 1024)
    n_layers = int(config.get("num_hidden_layers") or config.get("n_layer") or 12)
    vocab_size = int(config.get("vocab_size") or 32000)
    experts_per_token = int(
        config.get("num_experts_per_tok") or config.get("moe_top_k") or 2
    )
    expert_dim = int(
        config.get("moe_intermediate_size")
        or config.get("intermediate_size")
        or hidden_dim * 2
    )

    expert_ids = sorted(experts.keys())
    if max_experts is not None and len(expert_ids) > max_experts:
        print(f"[info] limiting experts {len(expert_ids)} → {max_experts}")
        expert_ids = expert_ids[:max_experts]

    n_experts = len(expert_ids) if expert_ids else int(
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("n_routed_experts")
        or 8
    )

    dense_blob = pack_dense(dense, quant=8)

    expert_blobs = []
    if expert_ids:
        for idx in expert_ids:
            blob = pack_one_expert(
                experts[idx], quant=quant, layout=layout,
                hidden_dim=hidden_dim, arch=arch,
            )
            expert_blobs.append(blob)
        expert_size = max(len(b) for b in expert_blobs)
        expert_blobs = [b + b"\0" * (expert_size - len(b)) for b in expert_blobs]
    else:
        print("[warn] no experts detected — creating placeholder square experts")
        expert_size = 4 + (
            hidden_dim * hidden_dim
            if quant == 8
            else hidden_dim * ((hidden_dim + 1) // 2)
        )
        expert_blobs = [b"\0" * expert_size for _ in range(n_experts)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        write_header(
            f,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            n_experts=n_experts,
            experts_per_token=experts_per_token,
            expert_dim=expert_dim,
            dense_size=len(dense_blob),
            expert_size=expert_size,
            dense_quant=8,
            expert_quant=quant,
            name=model_dir.name[:63],
        )
        f.write(dense_blob)
        for b in expert_blobs:
            f.write(b)

    total = out_path.stat().st_size
    print(f"[done] {out_path}")
    print(
        f"       dense={len(dense_blob)/1024/1024:.2f} MB  "
        f"experts={n_experts} x {expert_size/1024:.1f} KB"
    )
    print(f"       quant={quant}  layout={layout}  arch={arch}  total={total/1024/1024:.2f} MB")
    print(f"\nRun:  ./rabbit chat --model {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--model-dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-experts", type=int, default=None)
    ap.add_argument("--quant", type=int, choices=[4, 8], default=4, help="Expert quantization bits")
    ap.add_argument(
        "--layout",
        choices=["square", "flat"],
        default="square",
        help="square=force HxH matrix per expert (best for current kernels)",
    )
    ap.add_argument(
        "--arch",
        choices=["auto", "olmoe"],
        default="auto",
        help="Tensor classification: auto=heuristics, olmoe=exact map (docs/tensor-map.md)",
    )
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--n-experts", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    if args.demo:
        build_demo_model(out, hidden_dim=args.hidden_dim, n_experts=args.n_experts, quant=args.quant)
        return
    if not args.model_dir:
        ap.error("Need --demo or --model-dir")
    convert_hf(
        Path(args.model_dir), out, args.max_experts, args.quant, args.layout, args.arch
    )


if __name__ == "__main__":
    main()
