# Public MoE models

Realistic download + convert path for Rabbit. This replaces earlier draft notes that listed non-existent multi-hundred-B “Colibri roster” IDs.

## Roster

| Key | Hugging Face repo | Converter | Notes |
|-----|-------------------|-----------|--------|
| `tinybrainbot` | `nkthebass/tinybrainbot-360m-a75m-v4.1-moe-base` | `convert_hf_stream` | Primary live target (~360M Mixtral-style) |
| `olmoe` | `allenai/OLMoE-1B-7B-0924` | `convert_hf_to_rabbit --layout v2` | Real 1B-active / 7B-total MoE |
| `qwen-moe-a2.7b` | `Qwen/Qwen1.5-MoE-A2.7B` | `convert_hf_to_rabbit --layout v2` | gate/up/down experts |

```bash
pip install huggingface_hub safetensors numpy transformers

python tools/download_public_moe.py --list

# one model (default quant 4)
make public-moe MODEL=tinybrainbot

# or
python tools/download_public_moe.py --models olmoe --quant 4
```

## HF backend (optional)

```bash
pip install torch transformers
python tools/hf_backend.py --model-dir models/hf_public/tinybrainbot \
  --prompt "Hello" --max-new-tokens 8 --output /tmp/hf.json
```

Token-exact C validation against this JSON is tracked separately (validation harness PR).
