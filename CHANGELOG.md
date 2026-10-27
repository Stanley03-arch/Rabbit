## [0.4.10] - 2026-10-27

### Public MoE roster (realistic)
- `tools/download_public_moe.py` — TinyBrainBot, OLMoE-1B-7B, Qwen1.5-MoE-A2.7B
- Stream converter preferred for Mixtral-style; v2 fallback
- `make public-moe MODEL=…` / `make download-moe`
- `tools/hf_backend.py` optional HF greedy JSON helper
- Docs: `docs/PUBLIC_MOE.md`; offline `tests/test_public_moe_roster.py`

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
