# Phase D — Quality spine

## M1 — micro oracle (done)

```bash
make parity-v2
./rabbit chat --model models/ref-v2.rabbit   # forward: ref-v2
```

## M2 — HF-shaped faithful conversion (in progress)

### Done

- `--layout v2` packs **exact** embed / lm_head / rms / qkv / o / router and **exact** gate·up·down expert matrices (no H×H crop).
- Offline HF-like fixture with Qwen3-MoE tensor names (`make micro-hf`).
- `make m2` proves: fixture → convert → C greedy tokens match NumPy oracle.

```bash
make m2
./rabbit chat --model models/micro-hf-v2.rabbit
```

### Still open

| Item | Notes |
|------|--------|
| Per-layer experts | Engine still uses a **global** expert table (shared) |
| Live HF download | Point `--model-dir` at a real tiny MoE when you have one |
| transformers parity | Compare greedy tokens to HuggingFace for N steps |
| Tokenizer | Byte vocab only; real BPE later |
| Group-scaled int4 | After shapes are locked |

### Real checkpoint recipe

```bash
pip install safetensors numpy
python tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/hf_moe \
  --out models/real-v2.rabbit \
  --layout v2 --quant 8 --max-experts 8
./rabbit probe-v2 --model models/real-v2.rabbit --prompt hi --tokens 16
```

See [tensor-map.md](tensor-map.md).
