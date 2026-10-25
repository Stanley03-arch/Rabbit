#!/usr/bin/env python3
"""
Token parity harness for Phase D M2.

Always:
  C probe-v2 ↔ NumPy oracle on a Rabbit file (float32 or int8 experts).

Optional (when transformers + a checkpoint dir are available):
  HF greedy tokens vs Rabbit (requires matching architecture; most public
  tiny MoEs use MLA / non-standard layouts and will not match yet).

Usage:
  # Offline exact path (recommended CI):
  python tools/create_hf_micro_moe.py
  python tools/convert_hf_to_rabbit.py --model-dir models/hf_micro_moe \
      --out models/micro-hf-f32.rabbit --layout v2 --quant 0
  python tools/hf_token_parity.py --model models/micro-hf-f32.rabbit --prompt hi --tokens 16

  # Optional live HF (skips cleanly if unavailable):
  HF_MODEL=/path/to/hf_moe python tools/hf_token_parity.py --hf-dir $HF_MODEL ...
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from ref_forward_v2 import greedy_generate, load_ref_v2  # noqa: E402


def run_c(rabbit: Path, model: Path, prompt: str, n: int) -> list[int]:
    proc = subprocess.run(
        [str(rabbit), "probe-v2", "--model", str(model), "--prompt", prompt, "--tokens", str(n)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"probe-v2 failed:\n{out}")
    m = re.search(r"PROBE_V2.*ids=([0-9,]+)", out)
    if not m:
        raise RuntimeError(f"no PROBE_V2 line:\n{out}")
    return [int(x) for x in m.group(1).split(",") if x]


def try_hf_greedy(hf_dir: Path, prompt: str, n_new: int) -> list[int] | None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[hf] transformers/torch not installed — skip live HF compare")
        return None

    if not hf_dir.is_dir():
        print(f"[hf] missing {hf_dir} — skip")
        return None

    try:
        tok = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(hf_dir), trust_remote_code=True, torch_dtype=torch.float32
        )
        model.eval()
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=n_new,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        ids = out[0].tolist()
        print(f"[hf] generated {len(ids)} ids (prompt+new)")
        return ids
    except Exception as e:
        print(f"[hf] generate failed (arch mismatch expected for MLA/custom): {e}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/micro-hf-v2.rabbit")
    ap.add_argument("--prompt", default="hi")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--hf-dir", default=os.environ.get("HF_MODEL", ""))
    args = ap.parse_args()

    rabbit = ROOT / "rabbit"
    model_path = Path(args.model)
    if not rabbit.is_file():
        print("missing ./rabbit — run make", file=sys.stderr)
        return 1
    if not model_path.is_file():
        print(f"missing model {model_path}", file=sys.stderr)
        return 1

    py_ids = greedy_generate(load_ref_v2(model_path), args.prompt, args.tokens)
    c_ids = run_c(rabbit, model_path, args.prompt, args.tokens)

    if py_ids != c_ids:
        print("FAIL C ↔ NumPy", file=sys.stderr)
        print(" c", c_ids, file=sys.stderr)
        print("py", py_ids, file=sys.stderr)
        return 1

    print(f"ok C↔NumPy tokens={c_ids[:12]}...")

    if args.hf_dir:
        hf_ids = try_hf_greedy(Path(args.hf_dir), args.prompt, args.tokens)
        if hf_ids is not None:
            # Compare only suffix if HF includes prompt tokens
            print(f"[hf] ids head={hf_ids[:12]}...")
            print(
                "[hf] Note: token match requires identical tokenizer + arch; "
                "Rabbit micro uses byte ids. Use shared vocab for real parity."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
