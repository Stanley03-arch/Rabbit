# Converting models to Rabbit format

## Quick start (no download)

```bash
make demo-model
./rabbit chat --model models/demo-moe.rabbit
```

Creates an int4-quantized synthetic MoE.

## Real Hugging Face MoE

```bash
pip install safetensors numpy

python tools/convert_hf_to_rabbit.py \
    --model-dir /path/to/hf-moe \
    --out models/my-moe.rabbit \
    --quant 4 \
    --layout square \
    --max-experts 32

./rabbit chat --model models/my-moe.rabbit
```

### Flags

| Flag | Meaning |
|------|--------|
| `--quant 4\|8` | Expert weight bits (default 4) |
| `--layout square` | Force each expert to a `[H,H]` matrix (best match for current kernels) |
| `--layout flat` | Concatenate all expert tensors as-is |
| `--max-experts N` | Keep only first N experts (important for 4 GB RAM) |

## Expert on-disk layout (engine contract)

```
[ float32 scale  (4 bytes) ]
[ packed weights           ]   # int8 or int4 depending on header.expert_quant
```

- **int8**: one byte per weight, row-major `[out_dim x in_dim]`
- **int4**: two weights per byte (low nibble first), row-major

The engine reads `expert_quant` from the header and selects `gemv_int8` or `gemv_int4`.

## Tokenizer

Default is byte-level (vocab 256).

Optional word vocab:

```bash
# one token per line
./rabbit chat --model models/demo-moe.rabbit
# (future CLI flag will accept --vocab path; API already supports load)
```

API:

```c
rabbit_tokenizer_set_byte_mode();
rabbit_tokenizer_load_vocab("vocab.txt");
```

## Design notes for 4 GB machines

- Prefer `--quant 4`
- Prefer `--layout square` so kernels stay simple and dense
- Use `--max-experts` aggressively
- Dense part is still int8 in the converter; further dense quant is planned
