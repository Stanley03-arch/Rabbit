#!/usr/bin/env python3
"""
convert_hf_to_rabbit.py — CLI for HF → Rabbit conversion.

Expert quant: 0=f32, 4=group-int4 (group=32), 8=int8
Layouts: square | flat | v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rabbit_convert_pack import build_demo_model
from rabbit_convert_v2 import convert_hf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--model-dir", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max-experts", type=int, default=None)
    ap.add_argument("--quant", type=int, default=4, choices=[0, 4, 8])
    ap.add_argument("--layout", choices=["square", "flat", "v2"], default="square")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--n-experts", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    if args.demo:
        build_demo_model(out, hidden_dim=args.hidden_dim, n_experts=args.n_experts, quant=args.quant)
        return
    if not args.model_dir:
        ap.error("Need --demo or --model-dir")
    convert_hf(Path(args.model_dir), out, args.max_experts, args.quant, args.layout)


if __name__ == "__main__":
    main()
