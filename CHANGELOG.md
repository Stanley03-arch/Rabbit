## [0.4.12] - 2026-11-05

### CLI
- **Implemented** the `bench` command (was documented in 0.4.11 but missing from binary)
  - Times ref-v2 greedy generation (or falls back to probe path)
  - Reports average ms and tokens/sec across `--runs` (default 5)
  - Supports `--model`, `--prompt`, `--tokens`, `--tokenizer`, `--runs`
- Expanded help text with all commands and clearer examples
- Consistent option handling across chat / probe / probe-v2 / bench / stats

### Engine robustness
- Clearer model-load errors: ENOENT / EACCES / strerror with path context
- Generation paths now use `rabbit_xmalloc` / `rabbit_xcalloc` throughout

### Product surface
- `rabbit_server.py` and `rabbit_dashboard.py` both accept `--tokenizer` and pass it to the engine

### Docs & polish
- Version bumped to 0.4.12
- README and ROADMAP updated for current feature set

## [0.4.11] - 2026-11-03

### CLI
- New `bench` command: time ref-v2 greedy (or fallback probe) over multiple runs; reports avg ms and tokens/sec
- Help text includes bench examples and `--runs`

### Robustness
- Clearer model load errors (`ENOENT`, `EACCES`, strerror fallback)
- Generation paths use `rabbit_xmalloc` / `rabbit_xcalloc` instead of unchecked malloc

## [0.4.10] - 2026-10-28

### Utility helpers
- `rabbit_xcalloc`, `rabbit_xrealloc`, `rabbit_xstrdup`
- Formatted fatal helper `rabbit_dief`
- Declarations added to `rabbit.h`
- `main.c` now uses the safer alloc helpers

### CLI / docs
- Help text includes quick examples
- CONTRIBUTING updated with current helpers, `make check`, and live-moe note

## [0.4.9] - 2026-10-27

### Live MoE streaming harden
- `convert_hf_stream.py`: multi-shard safetensors via weight_map index, on-demand tensor load
- Support `--quant 0` (f32) and `--quant 4` (group-int4)
- Optional `lm_head.weight` when not tied to embed
- `make live-moe`: dependency check, download index.json + all shards, `LIVE_QUANT` override
- Docs: PHASE_D / ROADMAP updated for live MoE recipe

## [0.4.8] - 2026-10-26

### Fix f32 expert path (quant=0)
- `expert_quant == 0` was treated as default int8 (`?: 8`), breaking ref-v2 / exact-f32 parity
- Display shows `f32` when quant is 0

# Changelog

## [0.4.7] - 2026-10-25

### Final RMSNorm
- Dense layout: `embed | lm_head | final_norm | layers…`
- C `forward_v2` + NumPy oracle apply `model.norm` before logits
- Fixtures / converter pack `model.norm.weight`

## [0.4.6] - 2026-10-25

### GQA + Mixtral path
- Separate Q/K/V, header GQA metadata, Mixtral w1/w3/w2, `make mixtral-micro`

## [0.4.5] - 2026-10-25

### Group-scaled int4
- group=32 pack/dequant/GEMV + converter `--quant 4`

## [0.4.4] - 2026-10-25

### Real BPE tokenizer (RBTTOK01)

## [0.4.3] — exact f32 experts

## [0.4.2] — per-layer expert tables

## [0.4.1] — Phase D M1/M2 start

## [0.4.0] — OpenAI API + SSE
