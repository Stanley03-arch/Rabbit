# Rabbit 🐇

**Tiny engine. Immense models.**  
Run large Mixture-of-Experts (MoE) models on extremely constrained hardware (target: 4 GB RAM laptops) by treating disk as a first-class memory tier.

Inspired by [Colibrì](https://github.com/JustVugg/colibri).

> **Status**: Early foundation. Not production-ready. Work in progress.

## Goal

- Keep only the dense / shared parts of an MoE model in RAM
- Stream routed experts from disk on demand
- Extreme quantization support
- Pure C core, minimal dependencies
- Target machines with as little as **4 GB total RAM**

## Current Reality Check

Full trillion-scale models (Kimi K3 2.8T, large DeepSeek variants) are extremely challenging on 4 GB.  
Rabbit starts with the same architecture as Colibrì and aims for the densest possible resident set + aggressive expert streaming.

## Quick Start (coming soon)

```bash
# Build
make

# Run (once models are supported)
./rabbit chat --model models/your-model
```

## Project Structure

```
rabbit/
├── src/           # Core engine (C)
├── models/        # Model weights (user-provided)
├── tools/         # Conversion & utility scripts
├── docs/          # Architecture notes
└── Makefile
```

## License

Apache-2.0

---

Built with the same spirit as Colibrì: frontier models should not be locked behind datacenter walls.
