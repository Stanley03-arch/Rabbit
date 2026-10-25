# Rabbit 🐇

**Tiny engine. Immense models.**  
Run large Mixture-of-Experts (MoE) models on extremely constrained hardware (target: 4 GB RAM laptops) by treating disk as a first-class memory tier.

Inspired by [Colibrì](https://github.com/JustVugg/colibri).

> **Status**: Early foundation. Loader + expert mmap + tiny LRU cache working. Generation still a stub.

## Goal

- Keep only the dense / shared parts of an MoE model in RAM
- Stream routed experts from disk on demand via `mmap`
- Extreme quantization support
- Pure C core, minimal dependencies
- Target machines with as little as **4 GB total RAM**

## Current Reality Check

Full trillion-scale models (Kimi K3 2.8T, large DeepSeek variants) are extremely challenging on 4 GB.  
Rabbit starts with the same architecture as Colibrì and aims for the densest possible resident set + aggressive expert streaming.

## Quick Start

```bash
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit

# Build the engine
make

# Create a tiny synthetic test model
make test-model

# Run it
./rabbit chat --model models/tiny-test.rabbit
```

You should see the model header information and a simple chat stub.

## Project Structure

```
rabbit/
├── src/                 # Core engine (C)
│   ├── rabbit.h         # Public API + file format
│   ├── main.c
│   ├── model.c          # Loader
│   ├── memory.c         # mmap + LRU expert cache
│   ├── moe.c            # Router + expert streaming (stub)
│   ├── quant.c
│   ├── tokenizer.c
│   └── utils.c
├── tools/
│   └── create_test_model.c
├── models/              # Place .rabbit files here
├── docs/
│   ├── architecture.md
│   └── file-format.md
└── Makefile
```

## What works today

- Rabbit file format (header + dense + experts)
- Loading dense weights into RAM
- Memory-mapping the expert section
- Tiny LRU expert cache (API ready)
- Synthetic test model generator

## Next steps

1. Implement actual expert fetch + matmul path
2. Add a minimal forward pass on the tiny test model
3. Aggressive quantization kernels
4. Support for real open MoE checkpoints

## License

Apache-2.0

---

Built with the same spirit as Colibrì: frontier models should not be locked behind datacenter walls.
