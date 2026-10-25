# Model conversion

Convert open Mixture-of-Experts checkpoints into the Rabbit binary format.

See also the detailed [tensor map](tensor-map.md) for Qwen3-MoE / Mixtral / DeepSeek-style naming.

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

### Options

| Option | Description |
|--------|-------------|
| `--quant {4,8}` | Expert weight bit-width (default: 4) |
| `--layout square` | Reshape each expert to `[hidden, hidden]` for current kernels (prefers `down_proj` / `w2`) |
| `--layout flat` | Concatenate expert tensors without reshaping |
| `--max-experts N` | Keep only the first N experts (recommended on low-RAM hosts) |

## Expert record layout

Each expert is stored as:

```
float32 scale
packed weights    # int8 or int4 according to header.expert_quant
```

Records are padded to a uniform size so the engine can index them as  
`base + expert_id * expert_size`.

## Low-memory guidance

For ~4 GB systems:

1. Prefer `--quant 4`
2. Prefer `--layout square`
3. Set `--max-experts` conservatively
4. Ensure the dense section remains within available RAM after OS overhead

## Limitations

- Expert tensor detection is heuristic (covers common DeepSeek / Qwen / Mixtral naming).
- Dense weights are packed with a simple absmax scheme; not all architectures are reconstructed exactly.
- Current kernels apply a single square matrix per expert; full gate/up/down (SwiGLU) is Phase B.
- Very large checkpoints may still be impractical without further pruning or quantization.
