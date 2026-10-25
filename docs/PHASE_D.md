# Phase D — Quality spine

## M1 — micro oracle (done)

```bash
make parity-v2
./rabbit chat --model models/ref-v2.rabbit   # forward: ref-v2
```

## M2 — HF-shaped faithful conversion

### Done

- `--layout v2` packs exact embed / lm_head / rms / Q,K,V,O / router and exact gate·up·down experts
- Offline HF-like fixture (`make micro-hf`)
- `make m2` / `make exact-f32`
- Per-layer expert tables (`L×E` records)
- Real BPE tokenizer (RBTTOK01)
- Group-scaled int4 (`--quant 4`, group=32)
- **GQA** dense layout: separate Q/K/V with `n_heads`, `n_kv_heads`, `head_dim` in header.reserved

### Live public MoE (in progress)

Target: **[nkthebass/tinybrainbot-360m-a75m-v4.1-moe-base](https://huggingface.co/nkthebass/tinybrainbot-360m-a75m-v4.1-moe-base)**

| Field | Value |
|-------|--------|
| Arch | MixtralForCausalLM (standard GQA + SwiGLU, **not MLA**) |
| Shape | H=768, L=12, heads=12, kv_heads=4, head_dim=64 |
| MoE | 24 experts × inter=512, top-4 |
| Vocab | 32k (SentencePiece / BPE) |

```bash
# Download (needs huggingface-cli or git-lfs)
mkdir -p models/tinybrainbot
huggingface-cli download nkthebass/tinybrainbot-360m-a75m-v4.1-moe-base \
  --local-dir models/tinybrainbot

# Convert (start with quant 0 for parity, or 4 for disk size)
python tools/convert_hf_to_rabbit.py \
  --model-dir models/tinybrainbot \
  --out models/tinybrainbot-v2.rabbit \
  --layout v2 --quant 0

# Tokenizer
python tools/export_hf_tokenizer.py \
  --model models/tinybrainbot \
  --out models/tinybrainbot.tok

./rabbit probe-v2 --model models/tinybrainbot-v2.rabbit \
  --tokenizer models/tinybrainbot.tok --prompt "Hello" --tokens 32
```

See also `make live-moe` (downloads + converts when HF CLI is available).

### Real checkpoint recipe (generic)

```bash
pip install safetensors numpy
python tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/hf_moe \
  --out models/real-v2.rabbit \
  --layout v2 --quant 0
./rabbit probe-v2 --model models/real-v2.rabbit --prompt hi --tokens 16
```
