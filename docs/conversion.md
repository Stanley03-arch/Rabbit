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

### In-memory converter

```bash
python3 tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/hf-model \
  --out models/model.rabbit \
  --quant 4 \
  --layout v2 \
  --max-experts 32
```

### Low-RAM streaming converter (preferred for large MoEs)

Loads tensors on demand from single-file or multi-shard safetensors.

```bash
python3 tools/convert_hf_stream.py \
  --model-dir /path/to/hf-model \
  --out models/model.rabbit \
  --quant 4 \
  --name my-moe
```

**Auto-detected expert schemes** (see `tools/rabbit_name_maps.py`):

| Family | Experts | Router |
|--------|---------|--------|
| Mixtral | `block_sparse_moe.experts.*.w1\|w2\|w3` | `block_sparse_moe.gate` |
| Qwen2/3-MoE | `mlp.experts.*.gate_proj\|up_proj\|down_proj` | `mlp.gate` |
| moe.* | `moe.experts.*` (proj or w1/w2/w3) | `moe.gate` / `moe.router` |
| sparse_moe | `sparse_moe.experts.*.w1\|w2\|w3` | `sparse_moe.gate` |

Offline checks:

```bash
make stream-mixtral   # Mixtral micro fixture
make stream-qwen      # Qwen micro fixture
```

### Options

| Option | Description |
|--------|-------------|
| `--quant {0,4,8}` | Expert quant: `0`=f32 (stream/v2), `4`=group-int4, `8`=int8 |
| `--layout square\|flat\|v2` | In-memory converter only; prefer `v2` for fidelity |
| `--max-experts N` | Keep only the first N experts (in-memory path) |
| `--name` | Header model name (streaming path) |

## Expert record layout (v2)

Each expert is stored as gate / up / down matrices (f32 or group-int4).

Records are padded to a uniform size so the engine can index them as  
`base + (layer * n_experts + expert_id) * expert_size`.

## Low-memory guidance

For ~4 GB systems:

1. Prefer `convert_hf_stream.py` over the in-memory path
2. Prefer `--quant 4`
3. Prefer `--layout v2` when using the in-memory converter
4. Ensure the dense section remains within available RAM after OS overhead

## Limitations

- Expert tensor detection covers common Mixtral / Qwen / DeepSeek-style names; unusual layouts may need a template in `rabbit_name_maps.py`.
- Dense reconstruction is exact only on the v2 path for supported attention/router tensors.
- Very large checkpoints may still be impractical without further pruning or quantization.
