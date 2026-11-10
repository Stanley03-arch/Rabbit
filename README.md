# Rabbit

**Local MoE inference for constrained hardware.**

Rabbit is a pure-C engine that keeps dense weights in RAM and streams routed experts from disk. Target: limited-memory machines (including ~4 GB class laptops).

Inspired by the *idea* behind [Colibrì](https://github.com/JustVugg/colibri) — not a clone of its product or UI.

> **Status:** v0.4.28 — Phase A–G. Incremental ref-v2 decode, mmap-free expert load, multi-model dashboard, SSE metrics, OpenAI `usage`. **Windows/WSL:** `RABBIT_NO_MMAP=1` (see `docs/WINDOWS.md`).

[![Dashboard in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stanley03-arch/Rabbit/blob/main/notebooks/rabbit_colab_dashboard.ipynb)

---

## Interfaces

| Interface | Description |
|-----------|-------------|
| **CLI** | `./rabbit chat --model …` (`--top-p`, `--seed`) |
| **Stats** | `./rabbit stats --model …` (JSON) |
| **Bench** | `./rabbit bench --model … [--json]` (latency + cache deltas) |
| **Warm** | `./rabbit warm --model …` (prime expert cache) |
| **HTTP API** | OpenAI-compatible + SSE streaming + `usage` |
| **Dashboard** | Dark cockpit: multi-model switcher, SSE, voice, metrics |
| **Gradio** | Simple browser chat (`tools/rabbit_ui.py`) |
| **Windows** | `rabbit.bat` menu · `start-dashboard.bat` one-click |

### OpenAI-compatible API

```bash
make && make test-model
pip install flask
python tools/rabbit_server.py --model models/tiny-test.rabbit --port 8080

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"rabbit","messages":[{"role":"user","content":"hello"}],"max_tokens":32}'
```

### Engine dashboard

```bash
make && make test-model
pip install flask
python tools/rabbit_dashboard.py --model models/tiny-test.rabbit
# open http://127.0.0.1:7860/  — use the model dropdown to switch .rabbit files
```

---

## Build

```bash
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit
make
make test-model
./rabbit chat --model models/tiny-test.rabbit --temp 0.8 --top-p 0.9 --seed 42
./rabbit stats --model models/tiny-test.rabbit
./rabbit bench --model models/ref-v2.rabbit --tokens 16 --runs 5 --json
./rabbit warm  --model models/ref-v2.rabbit --tokens 16
```

Environment knobs:
```bash
export RABBIT_MEM_BUDGET_MB=1024    # default 1536
export RABBIT_CACHE_SLOTS=32        # expert cache slots (max 64)
export RABBIT_REF_V2_CTX=64         # ref-v2 KV window (16–256; lower = less RAM)
export RABBIT_NO_MMAP=1             # pread experts when mmap/FUSE is flaky
```

---

## Features

- Dense RAM + expert `mmap` / **pread** + LRU cache with pin/eviction stats
- Multi-head attention with per-head RoPE → dense residual → MoE
- Incremental ref-v2 decode (`RabbitRefV2Ctx` prefill/step, sliding KV)
- Group-scaled int4 (group=32) + GQA metadata in header
- DeepSeek-style **shared experts** (header + forward)
- HF → `.rabbit` low-RAM **streaming** converter (Mixtral / Qwen / OLMoE / DeepSeek)
- OpenAI-compatible HTTP API with SSE + `usage`
- Dashboard: multi-model registry, SSE tok/s, voice STT/TTS
- `rabbit bench --json` / `rabbit warm` for CI and cache priming

---

## Convert HF MoE

```bash
pip install safetensors numpy

python tools/convert_hf_stream.py --model-dir /path/to/hf \
  --out models/m.rabbit --quant 4 --name my-moe

make stream-mixtral    # block_sparse_moe + w1/w2/w3
make stream-qwen       # mlp.experts + gate/up/down_proj
make stream-deepseek   # + shared_experts (E+S records)
```

Live public models:
```bash
pip install huggingface_hub safetensors numpy transformers
make live-moe                 # TinyBrainBot 360M → quant 4
make live-moe LIVE_QUANT=0    # f32 experts
make live-olmoe               # OLMoE-1B-7B (~14 GB HF → ~6 GB int4)
make live-olmoe LIVE_QUANT=0  # f32 experts (large disk)
```

See [docs/WINDOWS.md](docs/WINDOWS.md), [docs/PHASE_D.md](docs/PHASE_D.md), [ROADMAP.md](ROADMAP.md).

---

## Limitations

- Synthetic weights do not produce real language.
- Dashboard is **Rabbit’s own UI**, not Colibrì’s.
- Large real MoEs still need aggressive quant / expert limits on 4 GB hosts.
- Voice STT needs a Chromium browser and microphone permission.
- `make live-moe` needs enough free disk/RAM for the checkpoint; prefer local disk + `RABBIT_NO_MMAP=1` on FUSE.

## License

Apache-2.0
