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
- `make bench-smoke` target added and included in `make check` + CI
- **`rabbit stats` JSON** includes `n_shared_experts`, GQA fields, fixed quant=0

### Engine robustness
- Clearer model-load errors; safer alloc helpers; row-wise int4 GEMV
- **DeepSeek-style shared experts** (header + disk layout + forward)

### Converters
- Shared `rabbit_name_maps.py`; Mixtral / Qwen / DeepSeek / moe variants
- Offline tests: `stream-mixtral`, `stream-qwen`, `stream-deepseek`, `test-name-maps`

### Product surface
- Server / dashboard / Gradio `--tokenizer`; dashboard Clear chat

## [0.4.11] - 2026-11-03

### CLI
- New `bench` command (timing); help examples

### Robustness
- Clearer model load errors; `rabbit_xmalloc` / `rabbit_xcalloc` in generation

## [0.4.10] - 2026-10-28

### Utility helpers
- `rabbit_xcalloc`, `rabbit_xrealloc`, `rabbit_xstrdup`, `rabbit_dief`

## [0.4.9] - 2026-10-27

### Live MoE streaming harden
- Multi-shard safetensors; quant 0|4; `make live-moe`

## [0.4.8] - 2026-10-26

### Fix f32 expert path (quant=0)

## [0.4.7] - 2026-10-25

### Final RMSNorm

## [0.4.6] - 2026-10-25

### GQA + Mixtral path

## [0.4.5] - 2026-10-25

### Group-scaled int4

## [0.4.4] - 2026-10-25

### Real BPE tokenizer (RBTTOK01)

## [0.4.3] — exact f32 experts

## [0.4.2] — per-layer expert tables

## [0.4.1] — Phase D M1/M2 start

## [0.4.0] — OpenAI API + SSE
