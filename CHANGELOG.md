## [0.4.28] - 2026-11-11

### Multi-model dashboard registry
- Scans `models/**/*.rabbit` at dashboard start; pairs sibling `.tok` files automatically
- **GET `/api/models`**, **POST `/api/models/select`**, **POST `/api/models/rescan`**
- Sidebar **model dropdown** switches active weights + tokenizer; cache counters reset on switch
- Chat / SSE use the active model from registry (not only the CLI `--model`)

## [0.4.27] - 2026-11-11

### Phase F close + Phase G start
- **`rabbit bench --json`**: latency (avg/min/max), tokens/sec, cache hit/miss/insert/evict deltas
- **`rabbit warm`**: one-shot expert-cache warm-up from a prompt trace
- **`docs/WINDOWS.md`**: `RABBIT_NO_MMAP` / WSL guidance for large models
- OpenAI server: shared `estimate_tokens`; stream final chunk includes `usage`
- Phase F complete; Phase G (observability & API fidelity) seeded

## [0.4.26] - 2026-11-11

### Phase F — streaming metrics
- Server tracks **tok/s**, **avg tok/s**, last token count, cache hit/miss **deltas** and **hits/s** per request
- SSE `done` event carries `ms`, `tps`, `tokens_est`, `cache_hits_delta`, `cache_misses_delta`, `cache_hits_per_s`
- Dashboard: live tok/s while streaming; session panel shows avg speed + last tokens; cache panel shows Δ hits and hits/s
- `/api/info` and `/api/metrics` expose the new fields

## [0.4.25] - 2026-11-11

### Larger public MoE recipe
- **`make live-olmoe`**: allenai/OLMoE-1B-7B-0924 (7B total / ~1.3B active, 64 experts top-8)
- HF ~14 GB BF16 → ~6 GB int4 `.rabbit` via streaming converter (peak RAM ≈ largest tensor)
- OLMoE tensor names are Qwen-style `mlp.experts` (already supported)
- Curl fallback for incomplete shards; `RABBIT_NO_MMAP=1` + 64 cache slots recommended for chat
- Phase E marked complete; Phase F seeded

## [0.4.24] - 2026-11-09

### Mmap-free expert load
- **`RABBIT_NO_MMAP=1`**: skip mmap; load experts with `pread` into the expert cache
- Always keep an open model fd + `expert_file_offset` so experts work when mmap fails
- `rabbit_expert_pread` + `rabbit_fetch_expert` fallback; ref-v2 MoE uses cache-aware fetch
- Shared-expert pinning works on the pread path
- Survives flaky FUSE/CDN mounts better (copy model to local disk still recommended)

## [0.4.23] - 2026-11-09

### Incremental ref-v2 decode
- **`RabbitRefV2Ctx`**: prefill + step API reuses per-layer KV across tokens (no O(n²) re-prefill)
- **`generate_v2` / `rabbit_ref_v2_greedy`** use the incremental path
- **Sliding-window KV**: when `RABBIT_REF_V2_CTX` is full, drop oldest slot and continue decode
- Parity with NumPy oracle unchanged (`make parity-v2`)

## [0.4.22] - 2026-11-09

### Ref-v2 memory + dashboard
- **`RABBIT_REF_V2_CTX=N`** (16–256): shrink per-layer KV cache; long prompts keep the last N tokens
- Dashboard expert-cache panel: **evictions**, **bytes copied**, **pinned** slots

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

## [0.4.15] - 2026-11-08

### Dashboard / GUI
- **Multiline composer** (Enter send, Shift+Enter newline) + char/token counter
- **Prompt library** (save chips in localStorage)
- **Persist chat**, regenerate last, auto-scroll lock, connection heartbeat
- **Markdown export**, completion sound, compact UI, UI scale
- **System prompt presets** (Technical / Teacher / Brief / Rabbit)
- **Drag-and-drop** import of JSON/MD chat exports
- **Voice shortcuts**: Ctrl+M always, M/V when not typing
- **Right-side voice controls**: header Voice button, floating FAB, mic after Send
- **Toasts** for mic / export / clear; double-click bubble to copy; hide FAB option
- Keyboard: Ctrl+L clear, Ctrl+R regen, Ctrl+K focus, Esc stop

## [0.4.14] - 2026-11-08

### Dashboard / GUI
- **SSE streaming chat** via `POST /api/chat/stream`
- **System prompt**, top-p / seed controls, session metrics (`/api/metrics`)
- **Export / import** chat JSON
- **Voice input** (Web Speech API) + **TTS** for replies
- **Voice command shortcuts**: clear, stop, mute, speak on/off, stream on/off, copy, export, refresh, repeat, send, help
- Expert grid shows **shared experts**; GQA + tokenizer in sidebar

### Windows
- **`rabbit.bat`** — interactive menu (build, chat, stats, dashboard, Gradio)
- **`start-dashboard.bat`** — one-click dashboard (builds model if missing)

### Gradio
- Top-p, seed, live stats panel, example prompts

## [0.4.13] - 2026-11-06

### Sampling
- **Nucleus sampling** via `--top-p` / `top_p` (default `1.0` = disabled)
- **Reproducible RNG** via `--seed` or `RABBIT_SEED` env
- New API: `rabbit_chat_ex2(model, max_tokens, temperature, top_p, seed)`

### HTTP API
- **CORS** headers on all responses (browser clients)
- **`POST /v1/completions`** — classic prompt-based text completion
- **`GET /health`** includes version, expert counts, shared experts, endpoint list
- Chat completions forward `top_p` and `seed` to the engine

### CLI
```bash
./rabbit chat --model m.rabbit --temp 0.8 --top-p 0.9 --seed 42
```

## [0.4.12] - 2026-11-05

### CLI
- **Implemented** the `bench` command (was documented in 0.4.11 but missing from binary)
- Expanded help text with all commands and clearer examples
