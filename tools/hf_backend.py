#!/usr/bin/env python3
"""
hf_backend.py — HF Transformers backend for validation harness.

Provides load_hf_model, hf_generate, hf_free functions that the C validation
code calls via a Python subprocess bridge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("pip install torch transformers accelerate", file=sys.stderr)
    sys.exit(1)


class HFBackend:
    """HF Transformers backend for greedy generation."""

    def __init__(self, model_dir: str, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load model and tokenizer."""
        print(f"[hf_backend] Loading {self.model_dir} on {self.device}", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float32,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        print(f"[hf_backend] Model loaded: {self.model.config.model_type}", file=sys.stderr)

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> tuple[list[int], list[float]]:
        """Generate tokens greedily and return (token_ids, logits_flat)."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]

        all_ids = input_ids[0].tolist()
        all_logits = []

        with torch.no_grad():
            for step in range(max_new_tokens):
                outputs = self.model(input_ids=input_ids, use_cache=True)
                logits = outputs.logits[0, -1, :]  # last token logits [vocab]
                all_logits.extend(logits.cpu().numpy().astype(float).tolist())

                if temperature == 0.0:
                    next_id = int(logits.argmax().item())
                else:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_id = int(torch.multinomial(probs, 1).item())

                all_ids.append(next_id)
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=input_ids.device)], dim=1)

                if next_id == self.tokenizer.eos_token_id:
                    break

        # Return only newly generated tokens (excluding prompt)
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = all_ids[prompt_len:]
        vocab_size = self.model.config.vocab_size
        new_logits = all_logits  # flat list: [step * vocab_size + v]

        return new_ids, new_logits

    def free(self) -> None:
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description="HF backend for Rabbit validation")
    ap.add_argument("--model-dir", required=True, help="HF model directory")
    ap.add_argument("--prompt", required=True, help="Prompt text")
    ap.add_argument("--max-new-tokens", type=int, default=32, help="Max new tokens")
    ap.add_argument("--temperature", type=float, default=0.0, help="Temperature (0=greedy)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Device")
    ap.add_argument("--output", required=True, help="Output JSON file")
    args = ap.parse_args()

    backend = HFBackend(args.model_dir, args.device)
    try:
        backend.load()
        ids, logits = backend.generate(args.prompt, args.max_new_tokens, args.temperature)
        result = {
            "token_ids": ids,
            "logits": logits,
            "vocab_size": backend.model.config.vocab_size if backend.model else 0,
        }
        Path(args.output).write_text(json.dumps(result))
        print(f"[hf_backend] Generated {len(ids)} tokens", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"[hf_backend] Error: {e}", file=sys.stderr)
        return 1
    finally:
        backend.free()


if __name__ == "__main__":
    sys.exit(main())