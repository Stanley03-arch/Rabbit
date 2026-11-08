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
- Supports F32 / F16 / BF16 / integer dtypes; BF16 decoded without torch
- Micro fixture writers (`create_*_micro.py`) now emit `.safetensors` with a pure writer (no `safetensors` package required)
- Offline stream tests (Mixtral / Qwen / DeepSeek) no longer require the safetensors package

## [0.4.16] - 2026-11-08

### Live MoE / conversion
- **Harden `make live-moe`**: validate `model.safetensors` size after HF download; curl fallback when Xet leaves a 0-byte placeholder; clearer progress messages
- **Converter** (`convert_hf_stream.py`): early reject of empty/tiny safetensors files with actionable Xet/CDN guidance; warn when a shard is >500 MB (mmap may bus-error on <~3 GB free RAM hosts)
- ROADMAP note: live TinyBrainBot still needs a host with enough free memory for the 1.5 GB checkpoint
