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
- **RABBIT_MAX_CACHE_SLOTS** raised 8 → **64** (budget-driven; was hard-capped too low for real MoEs)
- **`RABBIT_CACHE_SLOTS=N`** env override for exact slot count
- **Pin shared experts** (DeepSeek-style always-on) at load so LRU never evicts them
- Pre-allocate all slot buffers at `rabbit_cache_set_expert_size` (no malloc during generation)
- Track **evictions** + **bytes_copied**; richer free-time stats and `rabbit_cache_stats_ex`
- New API: `rabbit_cache_pin(cache, expert_id, src)`

## [0.4.17] - 2026-11-08

### Conversion / low-RAM
- **Streaming safetensors reader** in `convert_hf_stream.py`: pure header-parse + seek/read — no full-file mmap
- Peak RAM ≈ largest single tensor (not the whole checkpoint); enables convert on ~1–2 GB free hosts
