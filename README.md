# Rabbit 🐇

**Tiny engine. Immense models.**  
Run large Mixture-of-Experts models on constrained hardware (target: **4 GB RAM**) by streaming experts from disk.

Inspired by [Colibrì](https://github.com/JustVugg/colibri).

> **Status**: v0.2.1-dev — CLI options, int4 experts, HF converter, tokenizer, grouped attention

## Quick Start

```bash
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit
make

make test-model
./rabbit chat --model models/tiny-test.rabbit

make demo-model
./rabbit chat --model models/demo-moe.rabbit --tokens 80 --temp 0.7

# Optional word vocab
./rabbit chat --model models/demo-moe.rabbit --vocab models/sample_vocab.txt
```

### Convert a real HF MoE

```bash
pip install safetensors numpy

python tools/convert_hf_to_rabbit.py \
    --model-dir /path/to/hf-moe \
    --out models/my-moe.rabbit \
    --quant 4 --layout square --max-experts 32

./rabbit chat --model models/my-moe.rabbit --tokens 128
```

## CLI

```
rabbit chat --model <path> [options]

  --model <path>   .rabbit model (required)
  --vocab <path>   word vocab file (one token per line)
  --tokens <n>     max new tokens (default 64)
  --temp <f>       temperature (default 0.85)
```

## What works

| Feature | Status |
|---------|--------|
| Dense RAM + expert mmap + LRU | ✅ |
| Attention (grouped) + dense FFN + MoE | ✅ |
| int4 / int8 expert GEMV | ✅ |
| HF → Rabbit converter | ✅ |
| Byte + word tokenizer | ✅ |
| CLI: vocab / tokens / temp | ✅ |
| Production LM quality | ❌ |

## Docs

- [Architecture](docs/architecture.md)
- [File format](docs/file-format.md)
- [Conversion](docs/conversion.md)

## License

Apache-2.0
