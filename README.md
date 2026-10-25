# Rabbit

**Local MoE inference for constrained hardware.**

Rabbit is a pure-C engine that keeps dense weights in RAM and streams routed experts from disk. Target: limited-memory machines (including ~4 GB class laptops).

Inspired by the *idea* behind [Colibrì](https://github.com/JustVugg/colibri) — not a clone of its product or UI.

> **Status:** v0.3.0 — Phase A complete, Phase B started. Exact C↔NumPy parity on the ref model. SwiGLU-style expert path + basic RoPE attention when data allows. Full multi-head + memory caps still in progress.

[![Dashboard in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stanley03-arch/Rabbit/blob/main/notebooks/rabbit_colab_dashboard.ipynb)

---

## Interfaces

| Interface | Description |
|-----------|-------------|
| **CLI** | `./rabbit chat --model …` |
| **Dashboard** | Dark cockpit: chat, model stats, expert cache, activity grid |
| **Gradio** | Simple browser chat (`tools/rabbit_ui.py`) |

### Engine dashboard (recommended UI)

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
```

---

## Features

- Dense RAM + expert `mmap` + LRU cache  
- Attention (basic RoPE) → dense residual → MoE  
- int4 / int8 expert GEMV + SwiGLU-style path when gate/up/down present  
- HF → `.rabbit` converter (Qwen / Mixtral / DeepSeek heuristics)  
- CLI + original web dashboard  
- Exact C vs NumPy parity on the frozen ref model  

---

## Convert HF MoE

```bash
pip install safetensors numpy
python tools/convert_hf_to_rabbit.py --model-dir /path/to/hf --out models/m.rabbit --quant 4
# for structured experts (Phase B SwiGLU path):
python tools/convert_hf_to_rabbit.py --model-dir /path/to/hf --out models/m.rabbit --quant 4 --layout flat
```

See [docs/tensor-map.md](docs/tensor-map.md) and [docs/conversion.md](docs/conversion.md).

---

## Limitations

- Synthetic weights do not produce real language.  
- Dashboard is **Rabbit’s own UI**, not Colibrì’s.  
- Large real MoEs still need aggressive quant / expert limits on 4 GB hosts.  
- Full multi-head attention and hard memory caps are still evolving (Phase B).  

## License

Apache-2.0
