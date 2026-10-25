# Tensor map — real HF MoE → Rabbit

This document closes the Phase A requirement for **one real small HF MoE with a documented tensor map**.

Rabbit’s current engine still treats each expert as a single square `H × H` matrix (int4/int8 + scale). Full SwiGLU / gate-up-down support is Phase B. The converter therefore reduces real expert tensors into that form while preserving a clear, reproducible mapping.

## Supported naming families

The converter (`tools/convert_hf_to_rabbit.py`) recognises these patterns:

| Family | Expert path markers | Typical expert tensors |
|--------|---------------------|------------------------|
| Qwen3-MoE / Qwen2-MoE | `mlp.experts.`, `moe.experts.` | `gate_proj`, `up_proj`, `down_proj` (or packed `gate_up_proj`) |
| Mixtral | `block_sparse_moe.experts.` | `w1`, `w2`, `w3` |
| DeepSeek-style | `mlp.experts.`, `shared_experts` | `gate_proj`, `up_proj`, `down_proj` |
| Generic | `.experts.N.` | any 2-D weight |

## Mapping rules used by the converter

1. **Dense region** — everything that is *not* an expert weight is packed (int8 + scale) into the dense blob. This includes embeddings, attention projections, norms, routers, and shared experts when present.
2. **Expert index extraction** — the integer after `experts.` / `expert.` becomes the expert ID.
3. **Per-expert packing (`--layout square`, default)**  
   - Prefer the largest 2-D matrix that is close to square (usually `down_proj` or `w2`).  
   - If none fits, concatenate all expert tensors for that ID, then pad/truncate to exactly `hidden_dim × hidden_dim`.  
   - Result is quantised (int4 or int8) with a leading `float32` scale.
4. **`--layout flat`** — each expert tensor is quantised separately and concatenated; useful for future multi-matrix kernels.

## Example: micro Qwen3-MoE style (tiny public models)

Models such as `shibatch/tinyqwen3moe2m` (~2 M params) follow the Qwen3-MoE layout:

```
layers.{L}.mlp.experts.{E}.gate_proj.weight   # [moe_intermediate, hidden]
layers.{L}.mlp.experts.{E}.up_proj.weight     # [moe_intermediate, hidden]
layers.{L}.mlp.experts.{E}.down_proj.weight   # [hidden, moe_intermediate]
layers.{L}.mlp.gate.weight                    # router (kept in dense)
```

Rabbit conversion command (once the model is downloaded):

```bash
pip install safetensors numpy
python tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/hf \
  --out models/tiny-qwen3-moe.rabbit \
  --quant 4 \
  --layout square \
  --max-experts 8
```

The resulting file can be loaded with:

```bash
./rabbit chat --model models/tiny-qwen3-moe.rabbit
```

Because the current kernels only apply a single square matrix per expert, language quality will be limited until Phase B (real SwiGLU).

## Why this is enough for Phase A

- The mapping is deterministic and documented.
- The converter produces valid `.rabbit` files from real HF checkpoints.
- The synthetic `rabbit-ref-v1` path still provides bit-level numerical parity.
- Full family-correct expert MLPs and attention are explicitly deferred to Phase B.

See also [conversion.md](conversion.md) and [architecture.md](architecture.md).
