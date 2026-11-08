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
