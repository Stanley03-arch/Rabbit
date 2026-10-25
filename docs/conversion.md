# Model conversion

Convert open Mixture-of-Experts checkpoints into the Rabbit binary format.

## Synthetic demo (no download)

```bash
make demo-model
# equivalent:
python3 tools/convert_hf_to_rabbit.py --demo --out models/demo-moe.rabbit --quant 4
```

## Hugging Face checkpoints

Requirements:

```bash
pip install safetensors numpy
```

The model directory must contain `config.json` and one or more `.safetensors` files.

```bash
python3 tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/hf-model \
  --out models/model.rabbit \
  --quant 4 \
  --layout square \
  --max-experts 32
```

### Architecture-specific maps

| `--arch` | Behaviour |
|----------|-----------|
| `auto` (default) | Heuristic detection (DeepSeek / Qwen / Mixtral-style names) |
| `olmoe` | **Exact** OLMoE tensor map — see [tensor-map.md](tensor-map.md) |

Example for a real small MoE:

```bash
python3 tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/OLMoE-1B-7B-0924 \
  --out models/olmoe.rabbit \
  --arch olmoe \
  --quant 4 \
  --layout square \
  --max-experts 32
```

### Options

| Option | Description |
|--------|-------------|
| `--quant {4,8}` | Expert weight bit-width (default: 4) |
| `--layout square` | Reshape each expert to `[hidden, hidden]` for current kernels |
| `--layout flat` | Concatenate expert tensors without reshaping |
| `--max-experts N` | Keep only the first N experts (recommended on low-RAM hosts) |
| `--arch {auto,olmoe}` | Tensor classification strategy |

## Expert record layout

Each expert is stored as:

```
float32 scale
packed weights    # int8 or int4 according to header.expert_quant
```

Records are padded to a uniform size so the engine can index them as  
`base + expert_id * expert_size`.

For OLMoE under `--arch olmoe`, gate/up/down projections are concatenated then
padded/truncated to `H×H` (Phase A kernel contract). Phase B will restore a real
SwiGLU layout.

## Low-memory guidance

For ~4 GB systems:

1. Prefer `--quant 4`
2. Prefer `--layout square`
3. Set `--max-experts` conservatively
4. Ensure the dense section remains within available RAM after OS overhead

## Limitations

- Without `--arch olmoe`, expert tensor detection is heuristic.
- Dense weights are packed with a simple absmax scheme; not all architectures are reconstructed exactly.
- Very large checkpoints may still be impractical without further pruning or quantization.
- Full HF forward parity requires Phase B (real attention + expert MLP).
