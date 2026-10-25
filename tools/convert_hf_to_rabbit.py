#!/usr/bin/env python3
"""
convert_hf_to_rabbit.py

Convert an open Mixture-of-Experts model (Hugging Face style) into the
Rabbit binary format so it can be streamed by the Rabbit engine.

Current status
--------------
- Supports a clean, documented conversion path for MoE models that expose
  experts as separate tensors.
- Writes a valid Rabbit file (header + dense blob + expert blob).
- Includes a "demo" mode that builds a larger synthetic MoE without needing
  a real multi-GB download (useful for testing the engine on low-RAM machines).

Usage examples
--------------
  # Demo model (no download required)
  python tools/convert_hf_to_rabbit.py --demo --out models/demo-moe.rabbit

  # Real model (when you have the weights locally)
  python tools/convert_hf_to_rabbit.py \
      --model-dir /path/to/hf-model \
      --out models/my-model.rabbit \
      --max-experts 64          # optional: limit experts for 4GB machines

Notes for 4 GB target
---------------------
- Prefer heavily quantized checkpoints (int4 / int8).
- Use --max-experts to keep only a subset of experts if needed.
- The dense part must fit in ~1.5–2 GB RAM.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Rabbit on-disk header (must match src/rabbit.h)
# ---------------------------------------------------------------------------

RABBIT_MAGIC = b"RABBIT01"
HEADER_FMT = (
    "8s"   # magic
    "I"    # version
    "I"    # header_size
    "I"    # n_layers
    "I"    # hidden_dim
    "I"    # vocab_size
    "I"    # n_experts
    "I"    # experts_per_token
    "I"    # expert_dim
    "Q"    # dense_size
    "Q"    # expert_size
    "Q"    # total_experts_size
    "I"    # dense_quant
    "I"    # expert_quant
    "64s"  # name
    "128s" # reserved
)
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 8 + 4 + 4 + 4 * 6 + 8 * 3 + 4 * 2 + 64 + 128


def write_header(
    f,
    *,
    n_layers: int,
    hidden_dim: int,
    vocab_size: int,
    n_experts: int,
    experts_per_token: int,
    expert_dim: int,
    dense_size: int,
    expert_size: int,
    dense_quant: int,
    expert_quant: int,
    name: str,
) -> None:
    total_experts_size = n_experts * expert_size
    name_bytes = name.encode("utf-8")[:63].ljust(64, b"\0")
    reserved = b"\0" * 128

    header = struct.pack(
        HEADER_FMT,
        RABBIT_MAGIC,
        1,                    # version
        HEADER_SIZE,
        n_layers,
        hidden_dim,
        vocab_size,
        n_experts,
        experts_per_token,
        expert_dim,
        dense_size,
        expert_size,
        total_experts_size,
        dense_quant,
        expert_quant,
        name_bytes,
        reserved,
    )
    f.write(header)


# ---------------------------------------------------------------------------
# Demo model (no external weights required)
# ---------------------------------------------------------------------------

def build_demo_model(out_path: Path, hidden_dim: int = 128, n_experts: int = 16) -> None:
    """Create a larger synthetic MoE that still fits comfortably on low RAM."""
    n_layers = 6
    vocab_size = 256
    experts_per_token = 2
    expert_dim = hidden_dim * 2

    # Dense blob: embeddings + fake attention/FFN weights
    dense = np.random.randn(hidden_dim * 512).astype(np.float16)
    dense_bytes = dense.tobytes()

    # Each expert: int8 matrix [hidden_dim, hidden_dim]
    expert_size = hidden_dim * hidden_dim  # int8
    experts = []
    for e in range(n_experts):
        mat = np.random.randint(-128, 127, size=(hidden_dim, hidden_dim), dtype=np.int8)
        # Make experts slightly different
        mat = (mat + e * 3).astype(np.int8)
        experts.append(mat.tobytes())

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
            dense_size=len(dense_bytes),
            expert_size=expert_size,
            dense_quant=16,
            expert_quant=8,
            name="demo-moe",
        )
        f.write(dense_bytes)
        for blob in experts:
            f.write(blob)

    total = out_path.stat().st_size
    print(f"[demo] wrote {out_path}")
    print(f"       layers={n_layers}  hidden={hidden_dim}  experts={n_experts}")
    print(f"       dense={len(dense_bytes)/1024:.1f} KB  one_expert={expert_size} B")
    print(f"       total file={total/1024:.1f} KB")


# ---------------------------------------------------------------------------
# Hugging Face style conversion (best-effort)
# ---------------------------------------------------------------------------

def load_hf_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(cfg_path) as f:
        return json.load(f)


def find_safetensors(model_dir: Path) -> List[Path]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        # fallback: single pytorch_model.bin is harder; we focus on safetensors
        raise FileNotFoundError(
            f"No .safetensors files found in {model_dir}. "
            "Convert the model to safetensors first or use --demo."
        )
    return files


def try_import_safetensors():
    try:
        from safetensors import safe_open
        return safe_open
    except ImportError:
        print(
            "[error] 'safetensors' package is required for real model conversion.\n"
            "        pip install safetensors numpy\n"
            "        Or use --demo to generate a synthetic model.",
            file=sys.stderr,
        )
        sys.exit(1)


def collect_tensors(model_dir: Path) -> Dict[str, np.ndarray]:
    """Load all tensors from safetensors shards into a dict of numpy arrays."""
    safe_open = try_import_safetensors()
    tensors: Dict[str, np.ndarray] = {}

    for path in find_safetensors(model_dir):
        print(f"[load] {path.name}")
        with safe_open(path, framework="np") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def classify_tensors(
    tensors: Dict[str, np.ndarray],
    config: dict,
) -> Tuple[Dict[str, np.ndarray], Dict[int, Dict[str, np.ndarray]]]:
    """
    Split tensors into:
      - dense: everything that should stay in RAM
      - experts: dict[expert_id] -> {weight_name: array}

    Heuristics cover common naming patterns used by DeepSeek / Qwen / Mixtral style MoEs.
    """
    dense: Dict[str, np.ndarray] = {}
    experts: Dict[int, Dict[str, np.ndarray]] = {}

    expert_keywords = (
        "experts.",
        "block_sparse_moe.experts",
        "mlp.experts",
        "moe.experts",
    )

    for name, arr in tensors.items():
        is_expert = any(k in name for k in expert_keywords)
        if not is_expert:
            dense[name] = arr
            continue

        # Try to extract expert index
        # Patterns: ...experts.3.w1.weight  or  ...experts.3.weight
        idx = None
        parts = name.replace("/", ".").split(".")
        for i, p in enumerate(parts):
            if p == "experts" and i + 1 < len(parts) and parts[i + 1].isdigit():
                idx = int(parts[i + 1])
                break

        if idx is None:
            # Could not parse — keep in dense to be safe
            dense[name] = arr
            continue

        if idx not in experts:
            experts[idx] = {}
        experts[idx][name] = arr

    return dense, experts


def pack_dense(dense: Dict[str, np.ndarray], quant: int = 8) -> bytes:
    """Concatenate dense tensors into one blob. Simple int8 quant by default."""
    pieces = []
    for name in sorted(dense.keys()):
        arr = dense[name]
        if arr.dtype in (np.float16, np.float32, np.bfloat16):
            # crude absmax quant to int8
            f = arr.astype(np.float32)
            scale = np.max(np.abs(f)) + 1e-8
            q = np.clip(np.round(f / scale * 127.0), -128, 127).astype(np.int8)
            pieces.append(q.tobytes())
        else:
            pieces.append(arr.tobytes())
    return b"".join(pieces)


def pack_expert(expert_tensors: Dict[str, np.ndarray], target_size: Optional[int] = None) -> bytes:
    """Pack one expert's tensors into a single blob (int8)."""
    pieces = []
    for name in sorted(expert_tensors.keys()):
        arr = expert_tensors[name]
        if arr.dtype in (np.float16, np.float32):
            f = arr.astype(np.float32)
            scale = np.max(np.abs(f)) + 1e-8
            q = np.clip(np.round(f / scale * 127.0), -128, 127).astype(np.int8)
            pieces.append(q.tobytes())
        else:
            pieces.append(arr.astype(np.int8).tobytes() if arr.dtype != np.int8 else arr.tobytes())
    blob = b"".join(pieces)
    if target_size is not None:
        if len(blob) < target_size:
            blob = blob + b"\0" * (target_size - len(blob))
        else:
            blob = blob[:target_size]
    return blob


def convert_hf(
    model_dir: Path,
    out_path: Path,
    max_experts: Optional[int] = None,
    experts_per_token: int = 2,
) -> None:
    config = load_hf_config(model_dir)
    print("[config]", {k: config.get(k) for k in (
        "model_type", "hidden_size", "num_hidden_layers",
        "num_local_experts", "num_experts_per_tok", "vocab_size",
        "intermediate_size", "moe_intermediate_size",
    ) if k in config})

    tensors = collect_tensors(model_dir)
    print(f"[info] loaded {len(tensors)} tensors")

    dense, experts = classify_tensors(tensors, config)
    print(f"[info] dense tensors: {len(dense)}  expert groups: {len(experts)}")

    if not experts:
        print(
            "[warn] No expert tensors detected with current heuristics.\n"
            "       Falling back to packing everything as dense (not ideal for MoE).",
            file=sys.stderr,
        )

    # Architecture hints from config
    hidden_dim = int(config.get("hidden_size") or config.get("n_embd") or 1024)
    n_layers = int(config.get("num_hidden_layers") or config.get("n_layer") or 12)
    vocab_size = int(config.get("vocab_size") or 32000)
    n_experts_cfg = int(
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("n_routed_experts")
        or len(experts)
        or 8
    )
    experts_per_token = int(
        config.get("num_experts_per_tok")
        or config.get("moe_top_k")
        or experts_per_token
    )
    expert_dim = int(
        config.get("moe_intermediate_size")
        or config.get("intermediate_size")
        or hidden_dim * 2
    )

    # Optionally limit number of experts for low-RAM machines
    expert_ids = sorted(experts.keys())
    if max_experts is not None and len(expert_ids) > max_experts:
        print(f"[info] limiting experts from {len(expert_ids)} to {max_experts}")
        expert_ids = expert_ids[:max_experts]

    n_experts = len(expert_ids) if expert_ids else n_experts_cfg

    dense_blob = pack_dense(dense, quant=8)

    # Determine a uniform expert size
    expert_blobs = []
    for idx in expert_ids:
        blob = pack_expert(experts[idx])
        expert_blobs.append(blob)

    if expert_blobs:
        expert_size = max(len(b) for b in expert_blobs)
        expert_blobs = [pack_expert(experts[idx], target_size=expert_size) for idx in expert_ids]
    else:
        # Create empty placeholder experts so the format stays valid
        expert_size = hidden_dim * hidden_dim
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
            expert_quant=8,
            name=model_dir.name[:63],
        )
        f.write(dense_blob)
        for blob in expert_blobs:
            f.write(blob)

    total = out_path.stat().st_size
    print(f"[done] wrote {out_path}")
    print(f"       dense={len(dense_blob)/1024/1024:.2f} MB")
    print(f"       experts={n_experts} x {expert_size/1024:.1f} KB")
    print(f"       total={total/1024/1024:.2f} MB")
    print()
    print("You can now run:")
    print(f"  ./rabbit chat --model {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert HF MoE → Rabbit format")
    ap.add_argument("--demo", action="store_true", help="Build a synthetic demo MoE (no download)")
    ap.add_argument("--model-dir", type=str, help="Path to Hugging Face model directory")
    ap.add_argument("--out", type=str, required=True, help="Output .rabbit file")
    ap.add_argument("--max-experts", type=int, default=None, help="Limit number of experts (for low RAM)")
    ap.add_argument("--hidden-dim", type=int, default=128, help="Hidden size for --demo")
    ap.add_argument("--n-experts", type=int, default=16, help="Number of experts for --demo")
    args = ap.parse_args()

    out_path = Path(args.out)

    if args.demo:
        build_demo_model(out_path, hidden_dim=args.hidden_dim, n_experts=args.n_experts)
        return

    if not args.model_dir:
        ap.error("Either --demo or --model-dir is required")

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"[error] model dir not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    convert_hf(model_dir, out_path, max_experts=args.max_experts)


if __name__ == "__main__":
    main()
