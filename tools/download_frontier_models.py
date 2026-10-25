#!/usr/bin/env python3
"""
download_frontier_models.py — Download and convert frontier MoE models to .rabbit format.

Supported models (matching Colibri's roster):
- GLM-5.2-int4-g64 (744B) — Z.ai/GLM-5.2-colibri-int4-g64
- OLMoE-1B-7B (7B) — allenai/OLMoE-1B-7B-0924
- DeepSeek V4 Flash (284B) — deepseek-ai/DeepSeek-V2-Lite

Note: Full 744B/284B models require massive storage. This script downloads
quantized versions suitable for constrained hardware testing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    print("pip install huggingface_hub safetensors", file=sys.stderr)
    sys.exit(1)

FRONTIER_MODELS = {
    "glm-5.2-int4-g64": {
        "repo_id": "Z.ai/GLM-5.2-colibri-int4-g64",
        "description": "GLM-5.2 744B MoE int4 g64 (Colibri quant)",
        "quant": 4,
        "layout": "v2",
        "max_experts": 64,  # Limit for testing
        "rabbit_name": "glm-5.2-int4-g64",
    },
    "olmoe-1b-7b": {
        "repo_id": "allenai/OLMoE-1B-7B-0924",
        "description": "OLMoE 1B active / 7B total MoE",
        "quant": 4,
        "layout": "v2",
        "max_experts": None,
        "rabbit_name": "olmoe-1b-7b",
    },
    "deepseek-v2-lite": {
        "repo_id": "deepseek-ai/DeepSeek-V2-Lite",
        "description": "DeepSeek V2 Lite MoE (16B total)",
        "quant": 4,
        "layout": "v2",
        "max_experts": None,
        "rabbit_name": "deepseek-v2-lite",
    },
}


def download_model(repo_id: str, local_dir: Path, token: str | None = None) -> Path:
    """Download model from HF Hub."""
    print(f"[download] {repo_id} → {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )
    return local_dir


def convert_to_rabbit(
    model_dir: Path,
    out_path: Path,
    quant: int = 4,
    layout: str = "v2",
    max_experts: int | None = None,
) -> Path:
    """Convert HF model to .rabbit format using existing conversion pipeline."""
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "convert_hf_to_rabbit.py"),
        "--model-dir", str(model_dir),
        "--out", str(out_path),
        "--quant", str(quant),
        "--layout", layout,
    ]
    if max_experts:
        cmd.extend(["--max-experts", str(max_experts)])

    print(f"[convert] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"[error] conversion failed:\n{result.stdout}\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Conversion failed with code {result.returncode}")

    print(result.stdout)
    if not out_path.is_file():
        raise RuntimeError(f"Output not created: {out_path}")

    return out_path


def verify_model(rabbit_binary: Path, model_path: Path) -> bool:
    """Quick verification that model loads and produces tokens."""
    import subprocess
    import re

    result = subprocess.run(
        [str(rabbit_binary), "probe-v2", "--model", str(model_path), "--prompt", "hi", "--tokens", "8"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"[verify] probe-v2 failed: {result.stderr}", file=sys.stderr)
        return False

    match = re.search(r"PROBE_V2.*ids=([0-9,]+)", result.stdout + result.stderr)
    if not match:
        print("[verify] no PROBE_V2 output", file=sys.stderr)
        return False

    ids = [int(x) for x in match.group(1).split(",") if x]
    print(f"[verify] OK — generated {len(ids)} tokens: {ids[:8]}...")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", choices=list(FRONTIER_MODELS.keys()) + ["all"],
                    default=["all"], help="Models to download/convert")
    ap.add_argument("--models-dir", default="models/hf_frontier", help="HF model cache directory")
    ap.add_argument("--out-dir", default="models", help="Output directory for .rabbit files")
    ap.add_argument("--token", default=None, help="HF token (or set HF_TOKEN env)")
    ap.add_argument("--rabbit-binary", default="./rabbit", help="Path to rabbit binary for verification")
    ap.add_argument("--skip-download", action="store_true", help="Skip download, use existing HF cache")
    ap.add_argument("--skip-convert", action="store_true", help="Skip conversion, only download")
    ap.add_argument("--skip-verify", action="store_true", help="Skip verification step")
    args = ap.parse_args()

    models_to_process = list(FRONTIER_MODELS.keys()) if "all" in args.models else args.models
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rabbit_binary = Path(args.rabbit_binary)
    if not args.skip_verify and not rabbit_binary.is_file():
        print(f"[warn] rabbit binary not found at {rabbit_binary}, skipping verification", file=sys.stderr)

    for model_key in models_to_process:
        info = FRONTIER_MODELS[model_key]
        print(f"\n{'='*60}")
        print(f"Processing: {model_key} — {info['description']}")
        print(f"{'='*60}")

        hf_dir = models_dir / model_key
        rabbit_path = out_dir / f"{info['rabbit_name']}.rabbit"

        # Download
        if not args.skip_download:
            try:
                download_model(info["repo_id"], hf_dir, args.token)
            except Exception as e:
                print(f"[error] Download failed for {model_key}: {e}", file=sys.stderr)
                continue

        # Convert
        if not args.skip_convert:
            try:
                convert_to_rabbit(
                    hf_dir,
                    rabbit_path,
                    quant=info["quant"],
                    layout=info["layout"],
                    max_experts=info["max_experts"],
                )
            except Exception as e:
                print(f"[error] Conversion failed for {model_key}: {e}", file=sys.stderr)
                continue

        # Verify
        if not args.skip_verify and rabbit_binary.is_file() and rabbit_path.is_file():
            if not verify_model(rabbit_binary, rabbit_path):
                print(f"[warn] Verification failed for {model_key}", file=sys.stderr)

    print(f"\n{'='*60}")
    print("Done. Generated .rabbit models:")
    for model_key in models_to_process:
        info = FRONTIER_MODELS[model_key]
        rabbit_path = out_dir / f"{info['rabbit_name']}.rabbit"
        if rabbit_path.is_file():
            size_mb = rabbit_path.stat().st_size / (1024 * 1024)
            print(f"  {rabbit_path} ({size_mb:.1f} MB)")
        else:
            print(f"  {rabbit_path} (NOT FOUND)")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Fatal: {e}", file=sys.stderr)
        sys.exit(1)