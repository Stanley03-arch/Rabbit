## [0.4.22] - 2026-11-09

### Ref-v2 memory + dashboard
- **`RABBIT_REF_V2_CTX=N`** (16–256): shrink per-layer KV cache; long prompts keep the last N tokens
- Dashboard expert-cache panel: **evictions**, **bytes copied**, **pinned** slots

## [0.4.21] - 2026-11-08

### Ref-v2 hardening
- **GQA-aware** `is_ref_v2_model` dense size check (matches `parse_weights` including final_norm)
- **`RABBIT_FORCE_LEGACY=1`** / **`RABBIT_FORCE_REF_V2=1`** env overrides
- **generate_v2** uses BPE `rabbit_tokenize` / `rabbit_detokenize`

## [0.4.20] - 2026-11-08

### Tokenizer + ref-v2 detection
- **export_hf_tokenizer.py**: pure `tokenizer.json` path (no `transformers` required)
- **is_ref_v2_model**: GQA-aware dense size check

## [0.4.19] - 2026-11-08

### Live public MoE
- TinyBrainBot 360M end-to-end verified (streaming convert + chat)
- Expert cache hit rate 99%+ with 32 slots
