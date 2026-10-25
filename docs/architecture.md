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

## Expert Fetch Path (implemented)

```
rabbit_fetch_expert(model, expert_id, promote)
    │
    ├─ 1. Check tiny RAM LRU cache  → hit? return pointer
    │
    ├─ 2. Zero-copy pointer from mmap region
    │
    └─ 3. Optionally promote into LRU cache
```

## MoE Step (implemented for test model)

```
rabbit_moe_step(model, hidden, hidden_dim)
    │
    ├─ Run simple router → scores
    ├─ Select top-k experts
    ├─ For each selected expert:
    │       fetch (cache or mmap)
    │       apply_expert_stub()
    └─ Return
```

The current `apply_expert_stub` is only a placeholder linear transform so we can prove the plumbing. Real matmul kernels come next.

## File Format

See `docs/file-format.md`.

## Current Status

- Project skeleton and public API exist
- Loader + dense RAM + expert mmap working
- Expert fetch + LRU cache working
- Minimal forward pass on synthetic model working
- Real generation, quantization, and model conversion still to do

## Honesty

Full Kimi K3 (2.8T) or large DeepSeek V4 models will be extremely slow (or currently impractical) on 4 GB.  
Rabbit first aims to prove the architecture on smaller / heavily quantized MoE models, then scale up.
