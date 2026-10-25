# Changelog

## [0.4.3] - 2026-10-25

### Exact parity path
- Converter `--quant 0` packs pure float32 gate/up/down (no scale/int)
- C `forward_v2` + NumPy oracle load f32 experts when `expert_quant==0`
- `make exact-f32` + `tools/hf_token_parity.py` (optional transformers)

## [0.4.2] - 2026-10-25

### Per-layer expert tables
- C: `experts_shared` / `n_expert_records` / `rabbit_expert_ptr_layer`
- Converter `--layout v2` packs **L×E** records

## [0.4.1] - 2026-10-25

### Phase D M1 / M2 start
- Faithful `ref-v2` + `parity-v2` + HF micro fixture + `make m2`

## [0.4.0] - 2026-10-25

- OpenAI API + SSE, `rabbit stats`, dashboard metrics

## [0.3.x] - 2026-10-25

- Memory budget, multi-head RoPE, SwiGLU path
