# Changelog

## [0.4.2] - 2026-10-25

### Per-layer expert tables
- C: `experts_shared` / `n_expert_records` / `rabbit_expert_global_id` / `rabbit_expert_ptr_layer`
- Converter `--layout v2` packs **L×E** records (not a shared pool)
- `create_ref_v2` / HF micro fixture emit distinct experts per layer
- NumPy oracle + `forward_v2.c` use layer-local expert ids

## [0.4.1] - 2026-10-25

### Phase D M1
- Faithful `ref-v2` C forward + `probe-v2` + `parity-v2` (CI)
- Chat auto-selects ref-v2 path

### Phase D M2
- `--layout v2` HF conversion (exact dense + gate/up/down, no square crop)
- `tools/create_hf_micro_moe.py` offline Qwen3-MoE-named fixture
- `make micro-hf` / `make m2` + CI

## [0.4.0] - 2026-10-25

- OpenAI API + SSE, `rabbit stats`, dashboard metrics
- Phase D design + NumPy oracle start

## [0.3.x] - 2026-10-25

- Memory budget, multi-head RoPE, SwiGLU path
