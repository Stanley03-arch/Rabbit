#!/usr/bin/env python3
"""
Export a HuggingFace BPE tokenizer to Rabbit's compact RBTTOK01 format.

Supports tokenizers that expose:
  - get_vocab()
  - backend merges (GPT-2 / Llama / Qwen style byte-level BPE)

Usage:
  python tools/export_hf_tokenizer.py --hf-dir /path/to/model --out models/model.tok
  python tools/export_hf_tokenizer.py --demo --out models/demo.tok
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def write_rbtok(
    out_path: Path,
    id_to_token: Dict[int, str],
    merges: List[Tuple[str, str]],
    unk_id: int = 0,
    bos_id: int = 1,
    eos_id: int = 2,
    pad_id: int = 3,
) -> None:
    vocab_size = max(id_to_token.keys()) + 1 if id_to_token else 0
    # Ensure every id has a string
    tokens: List[bytes] = []
    for i in range(vocab_size):
        s = id_to_token.get(i, f"<unk_{i}>")
        if isinstance(s, bytes):
            tokens.append(s)
        else:
            tokens.append(s.encode("utf-8"))

    # Build token -> id for merge resolution
    tok_to_id = {tokens[i]: i for i in range(vocab_size)}

    merge_rows: List[Tuple[int, int, int]] = []
    for rank, (a, b) in enumerate(merges):
        ab = (a if isinstance(a, bytes) else a.encode("utf-8"))
        bb = (b if isinstance(b, bytes) else b.encode("utf-8"))
        left = tok_to_id.get(ab)
        right = tok_to_id.get(bb)
        if left is None or right is None:
            continue
        merge_rows.append((left, right, rank))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(b"RBTTOK01")
        f.write(struct.pack("<III", 1, vocab_size, len(merge_rows)))
        f.write(struct.pack("<iiii", unk_id, bos_id, eos_id, pad_id))
        for t in tokens:
            if len(t) > 65535:
                t = t[:65535]
            f.write(struct.pack("<H", len(t)))
            f.write(t)
        for left, right, rank in merge_rows:
            f.write(struct.pack("<III", left, right, rank))

    print(
        f"[tok] wrote {out_path}  vocab={vocab_size}  merges={len(merge_rows)}  "
        f"unk={unk_id} bos={bos_id} eos={eos_id}"
    )


def export_demo(out_path: Path) -> None:
    """Tiny synthetic BPE for unit tests (byte alphabet + a few merges)."""
    id_to_token: Dict[int, str] = {}
    # specials
    id_to_token[0] = "<unk>"
    id_to_token[1] = "<s>"
    id_to_token[2] = "</s>"
    id_to_token[3] = "<pad>"
    # single bytes 0..255 as tokens 4..259
    for b in range(256):
        id_to_token[4 + b] = chr(b) if 32 <= b < 127 else f"<{b:02x}>"

    # a few merges: 'h'+'i' -> 'hi', 'h'+'e' -> 'he'
    # We store merge pairs as the token strings that already exist
    merges: List[Tuple[str, str]] = [
        (chr(ord("h")), chr(ord("i"))),
        (chr(ord("h")), chr(ord("e"))),
        ("hi", " "),  # may not resolve if space token differs — ok
    ]
    # Add merged tokens
    id_to_token[260] = "hi"
    id_to_token[261] = "he"

    write_rbtok(out_path, id_to_token, merges, unk_id=0, bos_id=1, eos_id=2, pad_id=3)


def export_hf(hf_dir: Path, out_path: Path) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("pip install transformers", file=sys.stderr)
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)
    vocab = tok.get_vocab()  # token_str -> id
    id_to_token = {i: t for t, i in vocab.items()}

    merges: List[Tuple[str, str]] = []
    # Try several places HF stores merges
    backend = getattr(tok, "backend_tokenizer", None)
    if backend is not None:
        model = getattr(backend, "model", None)
        if model is not None and hasattr(model, "get_merges"):
            raw = model.get_merges()
            for line in raw:
                # "a b" or tuple
                if isinstance(line, (list, tuple)) and len(line) == 2:
                    merges.append((str(line[0]), str(line[1])))
                elif isinstance(line, str) and " " in line:
                    a, b = line.split(" ", 1)
                    merges.append((a, b))

    if not merges:
        # Fallback: merges.txt next to the model
        merges_path = hf_dir / "merges.txt"
        if merges_path.is_file():
            for line in merges_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    merges.append((parts[0], parts[1]))

    unk = getattr(tok, "unk_token_id", None) or 0
    bos = getattr(tok, "bos_token_id", None)
    if bos is None:
        bos = getattr(tok, "cls_token_id", None) or 1
    eos = getattr(tok, "eos_token_id", None) or 2
    pad = getattr(tok, "pad_token_id", None)
    if pad is None:
        pad = eos

    write_rbtok(
        out_path,
        id_to_token,
        merges,
        unk_id=int(unk),
        bos_id=int(bos),
        eos_id=int(eos),
        pad_id=int(pad),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        export_demo(args.out)
        return 0
    if not args.hf_dir:
        print("--hf-dir or --demo required", file=sys.stderr)
        return 1
    export_hf(args.hf_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
