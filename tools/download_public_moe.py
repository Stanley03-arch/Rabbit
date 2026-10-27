#!/usr/bin/env python3
"""Download public MoE checkpoints and convert to .rabbit (realistic roster).

Default targets are models that actually exist on the Hub and fit Rabbit's
Mixtral-style / standard MoE conversion paths. Large checkpoints still need
disk and RAM; use --quant 4 and the streaming converter where possible.

Examples:
  python tools/download_public_moe.py --models tinybrainbot --skip-verify
  python tools/download_public_moe.py --models olmoe --quant 4
  make public-moe MODEL=tinybrainbot
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PUBLIC_MOE: Dict[str, Dict[str, Any]] = {
    "tinybrainbot": {
        "repo_id": "nkthebass/tinybrainbot-360m-a75m-v4.1-moe-base",
        "description": "TinyBrainBot ~360M Mixtral-style MoE (primary live target)",
        "converter": "stream",  # convert_hf_stream.py
        "quant_default": 4,
        "rabbit_name": "tinybrainbot-v2",
        "allow_patterns": [
            "config.json",
            "*.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
    },
    "olmoe": {
        "repo_id": "allenai/OLMoE-1B-7B-0924",
        "description": "OLMoE 1B active / 7B total (AllenAI)",
        "converter": "v2",  # convert_hf_to_rabbit --layout v2
        "quant_default": 4,
        "rabbit_name": "olmoe-1b-7b",
        "allow_patterns": [
            "config.json",
            "*.safetensors",
            "model.safetensors.index.json",
            "tokenizer*",
            "special_tokens_map.json",
        ],
    },
    "qwen-moe-a2.7b": {
        "repo_id": "Qwen/Qwen1.5-MoE-A2.7B",
        "description": "Qwen1.5-MoE-A2.7B (gate/up/down experts)",
        "converter": "v2",
        "quant_default": 4,
        "rabbit_name": "qwen1.5-moe-a2.7b",
        "allow_patterns": [
            "config.json",
            "*.safetensors",
            "model.safetensors.index.json",
            "tokenizer*",
            "vocab.json",
            "merges.txt",
        ],
    },
}


def _hub_download(repo_id: str, local_dir: Path, patterns: List[str], token: Optional[str]) -> None:
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id} → {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        allow_patterns=patterns,
        token=token,
    )


def _convert_stream(model_dir: Path, out: Path, quant: int) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "convert_hf_stream.py"),
        "--model-dir",
        str(model_dir),
        "--out",
        str(out),
        "--quant",
        str(quant),
    ]
    print("[convert-stream]", " ".join(cmd))
    subprocess.check_call(cmd)


def _convert_v2(model_dir: Path, out: Path, quant: int, max_experts: Optional[int]) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "convert_hf_to_rabbit.py"),
        "--model-dir",
        str(model_dir),
        "--out",
        str(out),
        "--layout",
        "v2",
        "--quant",
        str(quant),
    ]
    if max_experts is not None:
        cmd.extend(["--max-experts", str(max_experts)])
    print("[convert-v2]", " ".join(cmd))
    subprocess.check_call(cmd)


def _export_tok(model_dir: Path, out_tok: Path) -> None:
    script = Path(__file__).parent / "export_hf_tokenizer.py"
    if not script.is_file():
        return
    cmd = [sys.executable, str(script), "--hf-dir", str(model_dir), "--out", str(out_tok)]
    print("[tokenizer]", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[warn] tokenizer export failed: {e}", file=sys.stderr)


def _verify(rabbit_bin: Path, model_path: Path, tok_path: Optional[Path]) -> bool:
    if not rabbit_bin.is_file() or not model_path.is_file():
        return False
    cmd = [str(rabbit_bin), "probe-v2", "--model", str(model_path), "--prompt", "Hello", "--tokens", "8"]
    if tok_path and tok_path.is_file():
        cmd.extend(["--tokenizer", str(tok_path)])
    print("[verify]", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        print(out, file=sys.stderr)
        return False
    if "PROBE_V2" not in out:
        print("[verify] missing PROBE_V2 line", file=sys.stderr)
        return False
    m = re.search(r"ids=([0-9,]+)", out)
    if m:
        print(f"[verify] OK ids={m.group(1)[:64]}...")
    return True


def process_one(
    key: str,
    models_dir: Path,
    out_dir: Path,
    quant: int,
    token: Optional[str],
    rabbit_bin: Path,
    skip_download: bool,
    skip_convert: bool,
    skip_verify: bool,
    max_experts: Optional[int],
) -> bool:
    info = PUBLIC_MOE[key]
    hf_dir = models_dir / key
    rabbit_path = out_dir / f"{info['rabbit_name']}.rabbit"
    tok_path = out_dir / f"{info['rabbit_name']}.tok"

    print("\n" + "=" * 60)
    print(f"{key}: {info['description']}")
    print("=" * 60)

    if not skip_download:
        try:
            _hub_download(info["repo_id"], hf_dir, info["allow_patterns"], token)
        except Exception as e:
            print(f"[error] download failed: {e}", file=sys.stderr)
            return False
    elif not (hf_dir / "config.json").is_file():
        print(f"[error] no config.json under {hf_dir} (download first)", file=sys.stderr)
        return False

    if not skip_convert:
        try:
            q = quant if quant is not None else int(info["quant_default"])
            if info["converter"] == "stream":
                try:
                    _convert_stream(hf_dir, rabbit_path, q)
                except Exception as e:
                    print(f"[warn] stream convert failed ({e}); trying v2 path", file=sys.stderr)
                    _convert_v2(hf_dir, rabbit_path, q, max_experts)
            else:
                _convert_v2(hf_dir, rabbit_path, q, max_experts)
            _export_tok(hf_dir, tok_path)
        except Exception as e:
            print(f"[error] convert failed: {e}", file=sys.stderr)
            return False

    if not skip_verify:
        if not _verify(rabbit_bin, rabbit_path, tok_path if tok_path.is_file() else None):
            print(f"[warn] verify failed for {key}", file=sys.stderr)
            return False

    if rabbit_path.is_file():
        mb = rabbit_path.stat().st_size / 1e6
        print(f"[done] {rabbit_path} ({mb:.1f} MB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        nargs="+",
        choices=list(PUBLIC_MOE.keys()) + ["all"],
        default=["tinybrainbot"],
        help="Which models (default: tinybrainbot)",
    )
    ap.add_argument("--models-dir", default="models/hf_public", type=Path)
    ap.add_argument("--out-dir", default="models", type=Path)
    ap.add_argument("--quant", type=int, default=None, choices=[0, 4, 8])
    ap.add_argument("--max-experts", type=int, default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--rabbit-binary", default="./rabbit", type=Path)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-convert", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--list", action="store_true", help="List roster and exit")
    args = ap.parse_args()

    if args.list:
        for k, v in PUBLIC_MOE.items():
            print(f"{k:16} {v['repo_id']}")
            print(f"{'':16} {v['description']}")
        return 0

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("pip install huggingface_hub safetensors numpy", file=sys.stderr)
        return 1

    keys = list(PUBLIC_MOE.keys()) if "all" in args.models else list(args.models)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for key in keys:
        if process_one(
            key,
            args.models_dir,
            args.out_dir,
            args.quant if args.quant is not None else PUBLIC_MOE[key]["quant_default"],
            args.token,
            args.rabbit_binary,
            args.skip_download,
            args.skip_convert,
            args.skip_verify,
            args.max_experts,
        ):
            ok += 1

    print(f"\nFinished {ok}/{len(keys)} model(s)")
    return 0 if ok == len(keys) else 1


if __name__ == "__main__":
    raise SystemExit(main())
