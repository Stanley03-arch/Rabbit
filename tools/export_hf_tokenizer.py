#!/usr/bin/env python3
"""
Export a HuggingFace BPE tokenizer to Rabbit's compact RBTTOK01 format.

Supports:
  - Pure tokenizer.json path (no transformers required)
  - AutoTokenizer + backend merges when transformers is available
  - merges.txt fallback

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
from typing import Dict, List, Tuple


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
    tokens: List[bytes] = []
    for i in range(vocab_size):
        s = id_to_token.get(i, f"<unk_{i}>")
        if isinstance(s, bytes):
            tokens.append(s)
        else:
            tokens.append(s.encode("utf-8"))

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
    id_to_token: Dict[int, str] = {}
    id_to_token[0] = "<unk>"
    id_to_token[1] = "<s>"
    id_to_token[2] = "</s>"
    id_to_token[3] = "<pad>"
    for b in range(256):
        id_to_token[4 + b] = chr(b) if 32 <= b < 127 else f"<{b:02x}>"
    merges: List[Tuple[str, str]] = [
        (chr(ord("h")), chr(ord("i"))),
        (chr(ord("h")), chr(ord("e"))),
        ("hi", " "),
    ]
    id_to_token[260] = "hi"
    id_to_token[261] = "he"
    write_rbtok(out_path, id_to_token, merges, unk_id=0, bos_id=1, eos_id=2, pad_id=3)


def _export_from_tokenizer_json(hf_dir: Path, out_path: Path) -> bool:
    """Pure-JSON export from tokenizer.json (no transformers dependency)."""
    tj = hf_dir / "tokenizer.json"
    if not tj.is_file():
        return False
    data = json.loads(tj.read_text(encoding="utf-8"))
    model = data.get("model") or {}
    vocab_map = model.get("vocab") or {}
    if not vocab_map:
        return False

    id_to_token: Dict[int, str] = {}
    for tok, idx in vocab_map.items():
        id_to_token[int(idx)] = tok

    for at in data.get("added_tokens") or []:
        if isinstance(at, dict) and "id" in at and "content" in at:
            id_to_token[int(at["id"])] = str(at["content"])

    merges: List[Tuple[str, str]] = []
    for line in model.get("merges") or []:
        if isinstance(line, (list, tuple)) and len(line) >= 2:
            merges.append((str(line[0]), str(line[1])))
        elif isinstance(line, str):
            parts = line.split(" ", 1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))

    unk_id, bos_id, eos_id, pad_id = 0, 1, 2, 3
    cfg_path = hf_dir / "tokenizer_config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            inv = {v: k for k, v in id_to_token.items()}
            for key, dest in (
                ("unk_token", "unk"),
                ("bos_token", "bos"),
                ("eos_token", "eos"),
                ("pad_token", "pad"),
            ):
                name = cfg.get(key)
                if isinstance(name, dict):
                    name = name.get("content") or name.get("id")
                if isinstance(name, str) and name in inv:
                    if dest == "unk":
                        unk_id = inv[name]
                    elif dest == "bos":
                        bos_id = inv[name]
                    elif dest == "eos":
                        eos_id = inv[name]
                    elif dest == "pad":
                        pad_id = inv[name]
        except Exception:
            pass

    for fname in ("generation_config.json", "config.json"):
        cp = hf_dir / fname
        if not cp.is_file():
            continue
        try:
            cfg = json.loads(cp.read_text(encoding="utf-8"))
            if cfg.get("bos_token_id") is not None:
                bos_id = int(cfg["bos_token_id"])
            if cfg.get("eos_token_id") is not None:
                eos_id = int(cfg["eos_token_id"])
            if cfg.get("pad_token_id") is not None:
                pad_id = int(cfg["pad_token_id"])
            if cfg.get("unk_token_id") is not None:
                unk_id = int(cfg["unk_token_id"])
        except Exception:
            pass

    write_rbtok(out_path, id_to_token, merges, unk_id=unk_id, bos_id=bos_id,
                eos_id=eos_id, pad_id=pad_id)
    return True


def export_hf(hf_dir: Path, out_path: Path) -> None:
    if _export_from_tokenizer_json(hf_dir, out_path):
        return

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "No tokenizer.json found and transformers is not installed.\n"
            "  pip install transformers   OR   provide tokenizer.json in --hf-dir",
            file=sys.stderr,
        )
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)
    vocab = tok.get_vocab()
    id_to_token = {i: t for t, i in vocab.items()}

    merges: List[Tuple[str, str]] = []
    backend = getattr(tok, "backend_tokenizer", None)
    if backend is not None:
        model = getattr(backend, "model", None)
        if model is not None and hasattr(model, "get_merges"):
            try:
                raw = model.get_merges()
                for line in raw:
                    if isinstance(line, (list, tuple)) and len(line) == 2:
                        merges.append((str(line[0]), str(line[1])))
                    elif isinstance(line, str) and " " in line:
                        a, b = line.split(" ", 1)
                        merges.append((a, b))
            except Exception:
                pass

    if not merges:
        merges_path = hf_dir / "merges.txt"
        if merges_path.is_file():
            for line in merges_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    merges.append((parts[0], parts[1]))

    if not merges:
        tj = hf_dir / "tokenizer.json"
        if tj.is_file():
            try:
                data = json.loads(tj.read_text(encoding="utf-8"))
                raw = (data.get("model") or {}).get("merges") or []
                for line in raw:
                    if isinstance(line, (list, tuple)) and len(line) >= 2:
                        merges.append((str(line[0]), str(line[1])))
                    elif isinstance(line, str):
                        parts = line.split(" ", 1)
                        if len(parts) == 2:
                            merges.append((parts[0], parts[1]))
            except Exception as e:
                print(f"[tok] warning: could not parse merges from tokenizer.json: {e}",
                      file=sys.stderr)

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
