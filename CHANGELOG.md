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
