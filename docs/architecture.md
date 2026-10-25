# Rabbit Architecture Notes

## Goal

Run large Mixture-of-Experts models on machines with as little as **4 GB total RAM**.

This is significantly more aggressive than Colibrì (which targets ~10–25 GB resident).

## Core Idea (same as Colibrì)

1. **Dense / shared weights** (attention, embeddings, shared experts, norms) stay in RAM.
2. **Routed experts** live on disk and are streamed on demand via `mmap` + a tiny LRU cache.
3. Extreme quantization so the dense part fits in ~1.5–2 GB.

## Memory Budget Target (4 GB machine)

| Component              | Target size     |
|------------------------|-----------------|
| OS + runtime overhead  | ~1.0 – 1.5 GB   |
| Dense weights (int4+)  | ≤ 1.8 GB        |
| Tiny expert cache      | 100 – 300 MB    |
| KV cache + activations  | remaining       |

Anything larger must come from disk.

## Planned Components

- `memory.c`   — mmap + LRU expert cache
- `moe.c`      — router + expert fetch/compute
- `quant.c`    — int4 / more aggressive kernels
- `model.c`    — loader for Rabbit file format
- `tokenizer.c`— model-specific tokenizers

## File Format (to be defined)

A simple custom format is planned:

```
[header]
  magic
  version
  architecture metadata
  dense size
  expert size
  num experts
  ...
[dense blob]          // always loaded into RAM
[expert blob]         // memory-mapped, streamed
```

## Current Status

- Project skeleton and public API exist
- Loader, chat loop, and expert streaming are still stubs
- Next concrete steps:
  1. Define the Rabbit model file format
  2. Implement dense loader + mmap of experts
  3. Implement a minimal forward pass on a tiny test MoE

## Honesty

Full Kimi K3 (2.8T) or large DeepSeek V4 models will be extremely slow (or currently impractical) on 4 GB.  
Rabbit first aims to prove the architecture on smaller / heavily quantized MoE models, then scale up.
