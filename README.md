# Rabbit 🐇

**Tiny engine. Immense models.**  
Run large Mixture-of-Experts (MoE) models on extremely constrained hardware (target: 4 GB RAM laptops) by treating disk as a first-class memory tier.

Inspired by [Colibrì](https://github.com/JustVugg/colibri).

> **Status**: Full engine path + HuggingFace → Rabbit converter available.

## Goal

- Keep only the dense / shared parts of an MoE model in RAM
- Stream routed experts from disk on demand via `mmap`
- Extreme quantization support
- Pure C core, minimal dependencies
- Target machines with as little as **4 GB total RAM**

## Quick Start

```bash
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit

make
make test-model                    # tiny synthetic model
./rabbit chat --model models/tiny-test.rabbit
```

### Larger demo MoE (still no download)

```bash
make demo-model
./rabbit chat --model models/demo-moe.rabbit
```

### Convert a real Hugging Face MoE

```bash
pip install safetensors numpy

python tools/convert_hf_to_rabbit.py \
    --model-dir /path/to/hf-moe \
    --out models/my-moe.rabbit \
    --max-experts 32

./rabbit chat --model models/my-moe.rabbit
```

See [docs/conversion.md](docs/conversion.md) for details.

## What works today

| Feature                         | Status |
|---------------------------------|--------|
| Rabbit file format              | ✅     |
| Dense weights → RAM             | ✅     |
| Experts memory-mapped + LRU     | ✅     |
| Expert fetch + int8 matmul      | ✅     |
| Attention + KV cache            | ✅     |
| Dense residual path             | ✅     |
| Full layer step                 | ✅     |
| Tokenizer + sampling            | ✅     |
| HF → Rabbit converter           | ✅     |
| Demo MoE generator              | ✅     |
| Production-quality generation   | ❌     |

## Architecture

```
Token → hidden
         │
         ▼
   ┌────────────────┐
   │  Attention (KV)   │
   ├────────────────┤
   │  Dense residual   │
   ├────────────────┤
   │  MoE (streamed)   │
   └────────────────┘
         │
         ▼
      LM head → sample
```

Experts live on disk and are pulled in on demand. Only a tiny LRU cache stays in RAM.

## Project layout

```
src/           C engine
tools/
  create_test_model.c
  convert_hf_to_rabbit.py    # HF MoE → .rabbit
docs/
  architecture.md
  file-format.md
  conversion.md
```

## Next steps

1. Better int4 packing and fused kernels
2. More accurate expert layout for DeepSeek / Kimi / Qwen MoEs
3. Real tokenizer support
4. Further memory reductions for true 4 GB targets

## License

Apache-2.0
