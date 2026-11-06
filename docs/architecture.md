# Architecture

## Goal

Run Mixture-of-Experts models on machines with limited RAM by treating disk as a first-class tier in the memory hierarchy.

Target: usable operation on systems with on the order of **4 GB** total memory, accepting that large frontier checkpoints will be slow and may need expert limiting.

## Memory model

| Region | Location | Notes |
|--------|----------|--------|
| Dense / shared weights | RAM | Embeddings, attention projections, norms, optional shared experts |
| Routed experts | Disk (`mmap`) | Loaded on demand |
| Expert LRU | RAM | Small fixed number of hot experts (default up to 8 slots, budget-driven) |
| KV cache | RAM | Short context window for the current attention implementation |

Placement affects **speed**, not numerical contract: the same expert weights are used whether they come from cache or mmap.

## Execution path (per layer)

1. **Attention** — multi-head / GQA with per-head RoPE; update KV cache; scaled dot-product over recent positions  
2. **Dense residual** — lightweight FFN-style update from the dense blob  
3. **MoE** — router scores → top-k experts → fetch → int4/int8 (or f32) GEMV; optional always-on shared experts  

## Expert fetch

```
rabbit_fetch_expert(model, id, promote)
  1. LRU hit  → return RAM pointer
  2. Else zero-copy pointer into mmap region
  3. Optionally copy into LRU (promote=1)
```

Per-layer vs shared expert tables are selected from the header and total record count at load time. DeepSeek-style shared experts live in the same disk layout (E routed + S shared per layer when present).

## Quantization contract

Expert blob layout (group-scaled int4):

```
[group scales …][packed matrix]
```

- `expert_quant == 0`: f32  
- `expert_quant == 4`: group-scaled int4 (group size 32)  
- `expert_quant == 8`: int8, row-major  

The engine selects the GEMV kernel from the model header. Dense section is typically f32 in current converters.

## GQA and header metadata

Header `reserved` (little-endian u32):

| Offset | Field |
|--------|--------|
| 0–3 | n_heads |
| 4–7 | n_kv_heads |
| 8–11 | head_dim |
| 12–15 | n_shared_experts |

Zero values mean “infer / absent”.

## File format

See [file-format.md](file-format.md).

## Non-goals (current)

- GPU runtime in main (optional CUDA is a planned next step)  
- Bit-exact match to every upstream MoE architecture  
- Production-grade chat quality on synthetic weights  
