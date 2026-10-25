# Rabbit 🐇

**Tiny engine. Immense models.**  
Run large Mixture-of-Experts (MoE) models on extremely constrained hardware (target: 4 GB RAM laptops) by treating disk as a first-class memory tier.

Inspired by [Colibrì](https://github.com/JustVugg/colibri).

> **Status**: End-to-end generation loop working on the synthetic test model.

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
make test-model
./rabbit chat --model models/tiny-test.rabbit
```

Type a message. Rabbit will:
1. Tokenize the prompt (byte-level)
2. Run MoE layers (router → fetch experts → int8 matmul)
3. Sample new tokens and print them

This is still a synthetic model, so the text is not meaningful language — it proves the full engine path works.

## What works today

| Feature                         | Status |
|---------------------------------|--------|
| Rabbit file format              | ✅     |
| Dense weights → RAM             | ✅     |
| Experts memory-mapped           | ✅     |
| Tiny LRU expert cache           | ✅     |
| Expert fetch (cache + mmap)     | ✅     |
| Real expert matmul (int8 GEMV)  | ✅     |
| Byte-level tokenizer            | ✅     |
| Sampling loop (temperature)     | ✅     |
| End-to-end generation           | ✅     |
| Real model conversion           | ❌     |
| Real language model quality     | ❌     |

## Project Structure

```
src/
  rabbit.h
  main.c
  model.c           # Loader + chat + generation
  memory.c          # mmap + LRU cache
  moe.c             # Router + fetch + matmul
  quant.c           # int8 / int4 helpers
  tokenizer.c       # byte-level tokenizer
  utils.c
tools/
  create_test_model.c
```

## Next steps

1. Attention / dense path
2. Better quantization + fused kernels
3. Real tokenizer support (SentencePiece / tiktoken style)
4. Converter for open MoE models (DeepSeek Flash, Kimi, …)

## License

Apache-2.0
