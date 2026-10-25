# Rabbit 🐇

**Tiny engine. Immense models.**  
Run large Mixture-of-Experts models on constrained hardware (target: **4 GB RAM**) by streaming experts from disk.

Inspired by [Colibrì](https://github.com/JustVugg/colibri).

> **Status**: Engine + int4 path + HF converter + tokenizer layer (v0.2.0-dev)

## Quick Start

```bash
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit
make

# Tiny test model
make test-model
./rabbit chat --model models/tiny-test.rabbit

# Larger int4 demo MoE
make demo-model
./rabbit chat --model models/demo-moe.rabbit
```

### Convert a real HF MoE

```bash
pip install safetensors numpy

python tools/convert_hf_to_rabbit.py \
    --model-dir /path/to/hf-moe \
    --out models/my-moe.rabbit \
    --quant 4 --layout square --max-experts 32

./rabbit chat --model models/my-moe.rabbit
```

## What works

| Feature | Status |
|---------|--------|
| Dense in RAM + experts mmap + LRU | ✅ |
| Attention → Dense FFN → MoE layer | ✅ |
| int8 + **int4** expert matmul | ✅ |
| Expert layout `[scale][packed weights]` | ✅ |
| HF → Rabbit converter | ✅ |
| Byte + simple word tokenizer | ✅ |
| Sampling loop | ✅ |
| Production LM quality | ❌ |

## Architecture

```
Token → hidden
         │
   Attention (KV cache)
   Dense residual (RAM)
   MoE (experts streamed from disk, int4/int8 GEMV)
         │
      LM head → sample
```

## Docs

- [Architecture](docs/architecture.md)
- [File format](docs/file-format.md)
- [Conversion](docs/conversion.md)

## License

Apache-2.0
