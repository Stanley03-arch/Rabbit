## [0.4.21] - 2026-11-08

### Ref-v2 hardening
- **GQA-aware** `is_ref_v2_model` dense size check (matches `parse_weights` including final_norm)
- **`RABBIT_FORCE_LEGACY=1`** / **`RABBIT_FORCE_REF_V2=1`** env overrides
- **generate_v2** uses BPE `rabbit_tokenize` / `rabbit_detokenize` (no byte-as-token fallback unless BPE missing)

## [0.4.20] - 2026-11-08

### Tokenizer + ref-v2 detection
- **export_hf_tokenizer.py**: pure `tokenizer.json` path (no `transformers` required); reads `model.merges` (64k merges for TinyBrainBot)
- **is_ref_v2_model**: GQA-aware dense size check (Q/K/V/O with n_kv_heads × head_dim)

## [0.4.19] - 2026-11-08

### Live public MoE
- **TinyBrainBot 360M end-to-end verified**: curl download (1.53 GB) → streaming convert (int4, 485 MB) → probe + chat
- Peak convert RAM ≈ largest tensor; works on ~1 GB free hosts
- Expert cache hit rate **99%+** with `RABBIT_CACHE_SLOTS=32` on live MoE traffic
- Dashboard parses **evictions / bytes_copied / pinned** cache fields

## [0.4.18] - 2026-11-08

### Expert cache
- **RABBIT_MAX_CACHE_SLOTS** raised 8 → **64**
- **`RABBIT_CACHE_SLOTS=N`** env override
- Pin shared experts; pre-allocate slots; eviction/byte stats
