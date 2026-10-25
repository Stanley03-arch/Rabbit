# Phase D — Quality spine

## M1 — micro oracle (done)

```bash
make parity-v2
./rabbit chat --model models/ref-v2.rabbit   # forward: ref-v2
```

## M2 — HF-shaped faithful conversion

### Done

- `--layout v2` packs **exact** embed / lm_head / rms / qkv / o / router and **exact** gate·up·down expert matrices.
- Offline HF-like fixture (`make micro-hf`).
- `make m2` — int8 experts, C↔NumPy match.
- Per-layer expert tables (`L×E` records).
- **`--quant 0`** pure float32 experts for unquantized parity (`make exact-f32`).
- `tools/hf_token_parity.py` — always C↔NumPy; optional `HF_MODEL=` transformers path.

```bash
make m2
make exact-f32
python tools/hf_token_parity.py --model models/micro-hf-f32.rabbit --prompt hi --tokens 16
```

### Still open

| Item | Notes |
|------|--------|
| Live HF + matching arch | Public tiny MoEs often use MLA / custom code; need standard MHA+SwiGLU MoE |
| Real BPE tokenizer | Byte vocab only today |
| Group-scaled int4 | After shapes locked |

### Real checkpoint recipe

```bash
pip install safetensors numpy
python tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/hf_moe \
  --out models/real-v2.rabbit \
  --layout v2 --quant 0
./rabbit probe-v2 --model models/real-v2.rabbit --prompt hi --tokens 16
python tools/hf_token_parity.py --model models/real-v2.rabbit --hf-dir /path/to/hf_moe
```

See [tensor-map.md](tensor-map.md).
