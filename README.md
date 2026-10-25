# Rabbit 🐇

**Tiny engine. Immense models.**  
Run large Mixture-of-Experts (MoE) models on extremely constrained hardware (target: 4 GB RAM laptops) by treating disk as a first-class memory tier.

Inspired by [Colibrì](https://github.com/JustVugg/colibri).

> **Status**: Core MoE path working. Real expert matmul (int8 GEMV) is live on the test model.

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

Type any message. Rabbit will:
1. Run a simple router
2. Select top-k experts
3. Fetch them (RAM cache → mmap)
4. Apply a real int8 matrix-vector multiply
5. Print a fingerprint of the final hidden state

## What works today

| Feature                         | Status |
|---------------------------------|--------|
| Rabbit file format              | ✅     |
| Dense weights → RAM             | ✅     |
| Experts memory-mapped           | ✅     |
| Tiny LRU expert cache           | ✅     |
| Expert fetch (cache + mmap)     | ✅     |
| Real expert matmul (int8 GEMV)  | ✅     |
| Minimal MoE forward             | ✅     |
| int8 / int4 dequant helpers     | ✅     |
| Tokenizer + sampling            | ❌     |
| Real model conversion           | ❌     |

## Project Structure

```
src/
  rabbit.h          Public API + format
  main.c
  model.c           Loader + chat + forward
  memory.c          mmap + LRU cache
  moe.c             Router + fetch + matmul kernels
  quant.c           int8 / int4 helpers
  tokenizer.c
  utils.c
tools/
  create_test_model.c
```

## Next steps

1. Better quantization + fused kernels
2. Tokenizer + sampling loop
3. Converter for real open MoE models (DeepSeek Flash, Kimi, …)
4. Attention / dense path implementation

## License

Apache-2.0
