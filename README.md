# Rabbit

**Local MoE inference for constrained hardware.**

Rabbit is a pure-C engine that keeps dense weights in RAM and streams routed experts from disk. Target: limited-memory machines (including ~4 GB class laptops).

Inspired by the *idea* behind [Colibrì](https://github.com/JustVugg/colibri) — not a clone of its product or UI.

> **Status:** Phase B in progress (v0.3.0). SwiGLU experts + RoPE attention; Phase A parity still holds for square/ref models.

[![Dashboard in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stanley03-arch/Rabbit/blob/main/notebooks/rabbit_colab_dashboard.ipynb)

---

## Interfaces

| Interface | Description |
|-----------|-------------|
| **CLI** | `./rabbit chat --model …` · `probe` · `mem` |
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
./rabbit mem --model models/tiny-test.rabbit
```

Checks:

```bash
make check    # unit tests + smoke + C↔NumPy parity + OLMoE map
```

---

## Features

- Dense RAM + expert `mmap` + LRU cache  
- Multi-head RoPE attention + dense residual + MoE  
- SwiGLU experts (when `expert_dim > hidden_dim`) or square GEMV  
- int4 / int8 expert quant  
- HF → `.rabbit` converter (heuristic + exact OLMoE map)  
- Memory budget report for ~4 GB hosts  

Docs: [architecture](docs/architecture.md) · [tensor-map](docs/tensor-map.md) · [Phase B](docs/PHASE_B.md)

---

## Convert HF MoE

```bash
pip install safetensors numpy

python tools/convert_hf_to_rabbit.py --model-dir /path/to/hf --out models/m.rabbit --quant 4

python tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/OLMoE-1B-7B-0924 \
  --out models/olmoe.rabbit \
  --arch olmoe --quant 4 --max-experts 32
```

---

## Limitations

- Square-collapsed experts still used for Phase A / converter default.  
- Dense attention/FFN weights are not fully structured QKV projections yet.  
- Large real MoEs need aggressive quant / expert limits on 4 GB hosts.

## License

Apache-2.0
