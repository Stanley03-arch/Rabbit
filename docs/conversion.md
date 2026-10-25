# Converting models to Rabbit format

## Quick start (no download)

```bash
# Create a larger synthetic MoE for testing
python tools/convert_hf_to_rabbit.py --demo --out models/demo-moe.rabbit

./rabbit chat --model models/demo-moe.rabbit
```

## Real Hugging Face MoE models

1. Download a model that uses safetensors (recommended):

```bash
# Example (small MoE if available)
huggingface-cli download <org>/<model> --local-dir ./hf-models/my-moe
```

2. Convert:

```bash
pip install safetensors numpy

python tools/convert_hf_to_rabbit.py \
    --model-dir ./hf-models/my-moe \
    --out models/my-moe.rabbit \
    --max-experts 32          # optional, helps on 4 GB machines
```

3. Run:

```bash
./rabbit chat --model models/my-moe.rabbit
```

## What the converter does

1. Reads `config.json` for architecture metadata
2. Loads all `.safetensors` shards
3. Splits tensors into:
   - **dense** (embeddings, attention, shared weights) → one RAM blob
   - **experts** (routed expert weights) → separate blobs, later memory-mapped
4. Quantizes floating-point weights to int8 (simple absmax)
5. Writes a single `.rabbit` file matching the engine header format

## Limitations (current version)

- Heuristic expert detection (works for common DeepSeek / Qwen / Mixtral naming)
- Only int8 quantization for now
- Does not yet reconstruct exact original matmul layouts for every architecture
- Very large models (hundreds of GB) still need aggressive `--max-experts` and further quant work for 4 GB RAM

## Design goal for 4 GB machines

- Dense part should stay under ~1.5–2 GB after quantization
- Experts are streamed from disk; only a tiny LRU stays in RAM
- Use `--max-experts` to reduce the expert set when necessary

## Next improvements

- Native int4 packing
- Better expert layout matching the engine kernels
- Direct support for more MoE families (Kimi, DeepSeek V4 Flash, …)
- Optional expert pruning / distillation for ultra-low RAM
