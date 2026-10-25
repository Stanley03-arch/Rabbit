# Architecture

## Goal

Run Mixture-of-Experts models on machines with limited RAM by treating disk as a first-class tier in the memory hierarchy.

Target: usable operation on systems with on the order of **4 GB** total memory, accepting that large frontier checkpoints will be slow and may need expert limiting.

## Memory model

| Region | Location | Notes |
|--------|----------|--------|
| Dense / shared weights | RAM | Embeddings, attention projections, shared experts, norms |
| Routed experts | Disk (`mmap`) | Loaded on demand |
| Expert LRU | RAM | Small fixed number of hot experts (default 4 slots) |
| KV cache | RAM | Short context window for the current attention implementation |

Placement affects **speed**, not numerical contract: the same expert weights are used whether they come from cache or mmap.

## Execution path (per layer)

1. **Attention** — update KV cache; grouped scaled dot-product over recent positions  
2. **Dense residual** — lightweight FFN-style update from the dense blob  
3. **MoE** — router scores → top-k experts → fetch → int4/int8 GEMV  

## Expert fetch

```
rabbit_fetch_expert(model, id, promote)
  1. LRU hit  → return RAM pointer
  2. Else zero-copy pointer into mmap region
  3. Optionally copy into LRU (promote=1)
```

## Quantization contract

Expert blob layout:

```
[float32 scale][packed matrix]
```

- `expert_quant == 8`: int8, row-major  
- `expert_quant == 4`: int4, two values per byte (low nibble first)

The engine selects the GEMV kernel from the model header.

## File format

See [file-format.md](file-format.md).

## Non-goals (current)

- GPU runtime (CPU/`mmap` first)
- Bit-exact match to every upstream MoE architecture
- Production-grade chat quality on synthetic weights
