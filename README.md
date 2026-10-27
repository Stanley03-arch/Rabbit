# Rabbit

**Local MoE inference for constrained hardware.**

Rabbit is a pure-C engine that keeps dense weights in RAM and streams routed experts from disk. Target: limited-memory machines (including ~4 GB class laptops).

Inspired by the *idea* behind [Colibrì](https://github.com/JustVugg/colibri) — not a clone of its product or UI.

> **Status:** v0.4.9 — Phase A–C complete enough for product surface; Phase D quality spine mostly done (parity, GQA, Mixtral naming, streaming HF converter). OpenAI-compatible HTTP API with **SSE streaming**, `rabbit stats`, dashboard driven by engine JSON.

[![Dashboard in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stanley03-arch/Rabbit/blob/main/notebooks/rabbit_colab_dashboard.ipynb)

---

## Interfaces

| Interface | Description |
|-----------|-------------|
| **CLI** | `./rabbit chat --model …` |
| **Stats** | `./rabbit stats --model …` (JSON) |
| **HTTP API** | OpenAI-compatible + SSE streaming |
| **Dashboard** | Dark cockpit: chat, model/memory stats, expert cache |
| **Gradio** | Simple browser chat (`tools/rabbit_ui.py`) |

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

**Local:**
```bash
make && make test-model
pip install flask
python tools/rabbit_dashboard.py --model models/tiny-test.rabbit
# open http://127.0.0.1:7860/
```

---

## Build

```bash
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit
make
make test-model
./rabbit chat --model models/tiny-test.rabbit
./rabbit stats --model models/tiny-test.rabbit
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
- HF → `.rabbit` converter (`layout v2`) and **low-RAM streaming** converter  
- OpenAI-compatible HTTP API with SSE streaming  
- Dashboard metrics from `rabbit stats`  
- Exact C vs NumPy parity on frozen ref models  

---

## Convert HF MoE

```bash
pip install safetensors numpy

# In-memory (more naming variants)
python tools/convert_hf_to_rabbit.py --model-dir /path/to/hf \
  --out models/m.rabbit --layout v2 --quant 4

# Low-RAM streaming (Mixtral-style w1/w3/w2 + multi-shard)
python tools/convert_hf_stream.py --model-dir /path/to/hf \
  --out models/m.rabbit --quant 4
```

Live public Mixtral-style model (downloads TinyBrainBot):
```bash
pip install huggingface_hub safetensors numpy transformers
make live-moe                 # quant 4
make live-moe LIVE_QUANT=0    # f32 experts
```

See [docs/tensor-map.md](docs/tensor-map.md), [docs/conversion.md](docs/conversion.md), and [docs/PHASE_D.md](docs/PHASE_D.md).

---

## Limitations

- Synthetic weights do not produce real language.  
- Dashboard is **Rabbit’s own UI**, not Colibrì’s.  
- Large real MoEs still need aggressive quant / expert limits on 4 GB hosts.  
- Streaming reads engine stdout as tokens are flushed; banner filtering is best-effort.  

## License

Apache-2.0
