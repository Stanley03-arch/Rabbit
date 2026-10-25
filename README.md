# Rabbit

**Local MoE inference for constrained hardware.**

Rabbit is a small pure-C engine that runs Mixture-of-Experts models by keeping dense weights in RAM and streaming routed experts from disk. The design target is machines with limited memory (including ~4 GB class laptops).

Inspired by [Colibrì](https://github.com/JustVugg/colibri).

> **Status:** early development (v0.2.2). Pipeline works on synthetic models. Browser UI available via Gradio.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stanley03-arch/Rabbit/blob/main/notebooks/rabbit_colab_ui.ipynb)

---

## Two ways to interact

| Mode | What you get |
|------|----------------|
| **CLI** | Terminal chat (`./rabbit chat --model ...`) |
| **Web UI** | Browser chat panel (Gradio) — closer to a Colibrì-style demo |

### Colab (recommended to see the UI)

**Web UI notebook:**  
https://colab.research.google.com/github/Stanley03-arch/Rabbit/blob/main/notebooks/rabbit_colab_ui.ipynb

**Terminal smoke test only:**  
https://colab.research.google.com/github/Stanley03-arch/Rabbit/blob/main/notebooks/rabbit_colab_smoke.ipynb

### Local web UI

```bash
make && make test-model
pip install gradio
python tools/rabbit_ui.py --model models/tiny-test.rabbit
```

Add `--share` if you need a public Gradio link.

---

## Features

| Area | Capability |
|------|------------|
| Memory | Dense in RAM; experts mmap + small LRU |
| Compute | Attention → dense residual → MoE |
| Quantization | int8 / int4 expert GEMV |
| Conversion | Hugging Face safetensors → `.rabbit` |
| UI | CLI + Gradio browser chat |

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

## CLI

```
rabbit chat --model <path> [--tokens N] [--temp F] [--vocab path]
```

---

## Convert HF MoE

```bash
pip install safetensors numpy
python tools/convert_hf_to_rabbit.py --model-dir /path/to/hf --out models/m.rabbit --quant 4
```

See [docs/conversion.md](docs/conversion.md).

---

## Limitations

- Synthetic weights do **not** produce real language; they prove the engine runs.
- Colibrì-like **UI** is provided via Gradio; model quality is a separate track.
- Large real MoEs still need aggressive quant / expert limits on 4 GB machines.

---

## License

Apache-2.0
