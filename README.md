# Rabbit

**Local MoE inference for constrained hardware.**

Rabbit is a pure-C engine that keeps dense weights in RAM and streams routed experts from disk. Target: limited-memory machines (including ~4 GB class laptops).

Inspired by the *idea* behind [Colibrì](https://github.com/JustVugg/colibri) — not a clone of its product or UI.

> **Status:** v0.4.16 — Phase A–D quality spine + product surface. Dashboard with **SSE streaming**, **voice input/TTS**, **voice commands**, shared-expert metrics. OpenAI-compatible HTTP API, `rabbit stats` / `bench`. **Windows:** `rabbit.bat` and `start-dashboard.bat`.

[![Dashboard in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stanley03-arch/Rabbit/blob/main/notebooks/rabbit_colab_dashboard.ipynb)

---

## Interfaces

| Interface | Description |
|-----------|-------------|
| **CLI** | `./rabbit chat --model …` (`--top-p`, `--seed`) |
| **Stats** | `./rabbit stats --model …` (JSON) |
| **Bench** | `./rabbit bench --model …` (avg ms + tokens/sec) |
| **HTTP API** | OpenAI-compatible + SSE streaming |
| **Dashboard** | Dark cockpit: chat, SSE, voice, metrics, expert cache |
| **Gradio** | Simple browser chat (`tools/rabbit_ui.py`) |
| **Windows** | `rabbit.bat` menu · `start-dashboard.bat` one-click |

### OpenAI-compatible API

```bash
make && make test-model
pip install flask
python tools/rabbit_server.py --model models/tiny-test.rabbit --port 8080

# non-streaming
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rabbit","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'

# streaming (SSE)
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rabbit","messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stream":true}'
```

### Engine dashboard

**Colab:**  
https://colab.research.google.com/github/Stanley03-arch/Rabbit/blob/main/notebooks/rabbit_colab_dashboard.ipynb

**Local (Linux/macOS):**
```bash
make && make test-model
pip install flask
python tools/rabbit_dashboard.py --model models/tiny-test.rabbit
# open http://127.0.0.1:7860/
```

**Windows:**
```bat
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit
rabbit.bat
rem  or one-click dashboard:
start-dashboard.bat
```
Requires **gcc** (MinGW / MSYS2 / WinLibs) to build and **Python 3** for the UI. Voice works best in Chrome or Edge.

Dashboard extras: mic + TTS, voice commands (`clear`, `mute`, `help`, …), SSE stream toggle, system prompt, chat export/import.

---

## Build

```bash
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit
make
make test-model
./rabbit chat --model models/tiny-test.rabbit --temp 0.8 --top-p 0.9 --seed 42
./rabbit stats --model models/tiny-test.rabbit
./rabbit bench --model models/tiny-test.rabbit --tokens 16 --runs 5
```

Memory budget (optional):
```bash
export RABBIT_MEM_BUDGET_MB=1024   # default 1536
```

---

## Features

- Dense RAM + expert `mmap` + LRU cache  
- Multi-head attention with per-head RoPE → dense residual → MoE  
- int4 / int8 expert GEMV + SwiGLU-style path when gate/up/down present  
- Group-scaled int4 (group=32) + GQA metadata in header  
- DeepSeek-style **shared experts** (header + forward)  
- HF → `.rabbit` converter (`layout v2`) and **low-RAM streaming** converter  
- Auto-detects **Mixtral**, **Qwen**, and **DeepSeek shared experts**  
- OpenAI-compatible HTTP API with SSE streaming  
- Dashboard: SSE, voice STT/TTS, voice commands, metrics from `rabbit stats`  
- `rabbit bench` for quick throughput measurement  
- Exact C vs NumPy parity on frozen ref models  

---

## Convert HF MoE

```bash
pip install safetensors numpy

# In-memory (more naming variants)
python tools/convert_hf_to_rabbit.py --model-dir /path/to/hf \
  --out models/m.rabbit --layout v2 --quant 4

# Low-RAM streaming — auto-detects Mixtral / Qwen / DeepSeek shared experts
python tools/convert_hf_stream.py --model-dir /path/to/hf \
  --out models/m.rabbit --quant 4 --name my-moe

# Offline naming checks
make stream-mixtral    # block_sparse_moe + w1/w2/w3
make stream-qwen       # mlp.experts + gate/up/down_proj
make stream-deepseek   # + shared_experts (E+S records)
```

Live public Mixtral-style model (downloads TinyBrainBot):
```bash
pip install huggingface_hub safetensors numpy transformers
make live-moe                 # quant 4
make live-moe LIVE_QUANT=0    # f32 experts
```

Demo walkthrough video: [docs/media](docs/media/) (`bash docs/media/assemble_media.sh` after adding MP4 parts).

See [docs/tensor-map.md](docs/tensor-map.md), [docs/conversion.md](docs/conversion.md), and [docs/PHASE_D.md](docs/PHASE_D.md).

---

## Limitations

- Synthetic weights do not produce real language.  
- Dashboard is **Rabbit’s own UI**, not Colibrì’s.  
- Large real MoEs still need aggressive quant / expert limits on 4 GB hosts.  
- Streaming reads engine stdout as tokens are flushed; banner filtering is best-effort.  
- Voice STT needs a Chromium browser and microphone permission.  
- `make live-moe` needs ~3 GB free RAM for the 1.5 GB TinyBrainBot checkpoint (safetensors mmap).

## License

Apache-2.0
