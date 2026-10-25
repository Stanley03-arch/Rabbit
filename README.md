# Rabbit

**Local MoE inference for constrained hardware.**

Rabbit is a pure-C engine that keeps dense weights in RAM and streams routed experts from disk. Target: limited-memory machines (including ~4 GB class laptops).

Inspired by the *idea* behind [Colibrì](https://github.com/JustVugg/colibri) — not a clone of its product or UI.

> **Status:** Phase A complete (v0.2.5). Synthetic models validate the pipeline; real HF MoE tensor map documented for OLMoE. Original **engine dashboard** available.

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

Phase A checks:

```bash
make check    # unit tests + smoke + C↔NumPy parity
```

---

## Features

- Dense RAM + expert `mmap` + LRU cache  
- Attention → dense residual → MoE  
- int4 / int8 expert GEMV  
- HF → `.rabbit` converter (heuristic + **exact OLMoE map**)  
- CLI + original web dashboard  

Tensor maps and packing rules: [docs/tensor-map.md](docs/tensor-map.md).

---

## Convert HF MoE

```bash
pip install safetensors numpy

# Heuristic path (Mixtral / DeepSeek / Qwen-style)
python tools/convert_hf_to_rabbit.py --model-dir /path/to/hf --out models/m.rabbit --quant 4

# Exact OLMoE map (Phase A)
python tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/OLMoE-1B-7B-0924 \
  --out models/olmoe.rabbit \
  --arch olmoe --quant 4 --max-experts 32
```

---

## Limitations

- Synthetic / square-collapsed experts do not produce real language quality yet.  
- Dashboard is **Rabbit’s own UI**, not Colibrì’s.  
- Large real MoEs still need aggressive quant / expert limits on 4 GB hosts.  
- Full HF forward parity needs Phase B (real SwiGLU + RoPE attention).

## License

Apache-2.0
