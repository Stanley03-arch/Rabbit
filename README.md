# Rabbit 🐇

**Tiny engine. Immense models.**  
Run large Mixture-of-Experts (MoE) models on extremely constrained hardware (target: 4 GB RAM laptops) by treating disk as a first-class memory tier.

Inspired by [Colibrì](https://github.com/JustVugg/colibri).

> **Status**: Foundation working. Expert fetch + minimal forward pass on synthetic model is live.

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

# Build
make

# Create a tiny synthetic test model
make test-model

# Run
./rabbit chat --model models/tiny-test.rabbit
```

Type any message. Rabbit will run a real (tiny) MoE forward pass:
- router selects top-k experts
- experts are fetched from mmap (and promoted into a small LRU cache)
- a simple transformation is applied

This proves the core plumbing works on low-memory machines.

## What works today

| Feature                        | Status      |
|--------------------------------|-------------|
| Rabbit file format             | ✅          |
| Dense weights → RAM            | ✅          |
| Experts memory-mapped          | ✅          |
| Tiny LRU expert cache          | ✅          |
| Expert fetch (cache + mmap)    | ✅          |
| Minimal MoE forward (test model)| ✅         |
| Real generation / tokenizer    | ❌ stub     |
| Quantization kernels           | ❌ stub     |
| Real model conversion          | ❌ planned  |

## Project Structure

```
rabbit/
├── src/
│   ├── rabbit.h
│   ├── main.c
│   ├── model.c          # Loader + chat + tiny forward
│   ├── memory.c         # mmap + LRU cache
│   ├── moe.c            # Router + expert fetch + step
│   ├── quant.c
│   ├── tokenizer.c
│   └── utils.c
├── tools/
│   └── create_test_model.c
├── models/
├── docs/
│   ├── architecture.md
│   └── file-format.md
└── Makefile
```

## Next steps

1. Better expert compute kernels (real matmuls)
2. Quantization (int4 / lower)
3. Tokenizer + real sampling loop
4. Converter for open MoE checkpoints (DeepSeek Flash, Kimi, etc.)

## License

Apache-2.0

---

Built with the same spirit as Colibrì: frontier models should not be locked behind datacenter walls.
