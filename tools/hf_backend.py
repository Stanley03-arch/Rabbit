#!/usr/bin/env python3
"""HF Transformers greedy backend for optional token/logit checks.

Writes JSON: {token_ids, logits (flat last-step or all steps), vocab_size}.
Used by future `rabbit validate` and ad-hoc parity scripts — not required for convert.

  python tools/hf_backend.py --model-dir models/hf_public/tinybrainbot \
    --prompt Hello --max-new-tokens 8 --output /tmp/hf.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--logits-mode", choices=["last", "all"], default="last",
                    help="last=only final step logits; all=every step (large)")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("pip install torch transformers", file=sys.stderr)
        return 1

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"model dir not found: {model_dir}", file=sys.stderr)
        return 1

    print(f"[hf_backend] load {model_dir} on {args.device}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
        device_map=None if args.device == "cpu" else args.device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if args.device == "cpu":
        model = model.to("cpu")

    inputs = tok(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    prompt_len = int(input_ids.shape[1])
    all_logits: list[float] = []
    new_ids: list[int] = []

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            out = model(input_ids=input_ids, use_cache=True)
            logits = out.logits[0, -1, :]
            if args.logits_mode == "all":
                all_logits.extend(logits.float().cpu().tolist())
            else:
                all_logits = logits.float().cpu().tolist()

            if args.temperature <= 0.0:
                next_id = int(logits.argmax().item())
            else:
                probs = torch.softmax(logits / args.temperature, dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())

            new_ids.append(next_id)
            next_t = torch.tensor([[next_id]], device=input_ids.device)
            input_ids = torch.cat([input_ids, next_t], dim=1)
            if tok.eos_token_id is not None and next_id == tok.eos_token_id:
                break

    vocab = int(getattr(model.config, "vocab_size", 0) or 0)
    payload = {
        "token_ids": new_ids,
        "logits": all_logits,
        "vocab_size": vocab,
        "prompt_len": prompt_len,
        "logits_mode": args.logits_mode,
    }
    Path(args.output).write_text(json.dumps(payload))
    print(f"[hf_backend] wrote {args.output} tokens={len(new_ids)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
