# Rabbit

**Local MoE inference for constrained hardware.**

Rabbit is a small pure-C engine that runs Mixture-of-Experts models by keeping dense weights in RAM and streaming routed experts from disk. The design target is machines with limited memory (including ~4 GB class laptops).

Inspired by [Colibrì](https://github.com/JustVugg/colibri).

> **Status:** early development (v0.2.2). The full pipeline works on synthetic models. Real open MoE conversion is supported but still evolving.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Stanley03-arch/Rabbit/blob/main/notebooks/rabbit_colab_smoke.ipynb)

---

## Features

| Area | Capability |
|------|------------|
| Memory | Dense weights resident; experts memory-mapped and cached in a small LRU |
| Compute | Layer path: attention → dense residual → MoE |
| Quantization | int8 and int4 expert GEMV; converter defaults to int4 |
| Conversion | Hugging Face safetensors → `.rabbit` |
| Interface | Interactive chat CLI with temperature and token limits |
| Tokenizer | Byte-level (default) or simple word vocabulary file |

---

## Requirements

- C11 compiler (`gcc` or `clang`)
- POSIX (`mmap`)
- Optional: Python 3 + `numpy` + `safetensors` for model conversion

---

## Build

```bash
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit
make
```

```bash
make test-model
./rabbit chat --model models/tiny-test.rabbit

make demo-model   # optional larger int4 demo
make smoke        # non-interactive check
```

---

## Colab smoke test

Open the notebook:

**[notebooks/rabbit_colab_smoke.ipynb](https://colab.research.google.com/github/Stanley03-arch/Rabbit/blob/main/notebooks/rabbit_colab_smoke.ipynb)**

It clones the repo, builds, creates a test model, and runs a non-interactive generation pass with cache stats.

---

## CLI

```
rabbit version
rabbit chat --model <path> [options]

  --model <path>   Path to a .rabbit model (required)
  --vocab <path>   Word vocabulary file (one token per line)
  --tokens <n>     Maximum new tokens (default: 64)
  --temp <f>       Sampling temperature (default: 0.85)
```

---

## Convert a Hugging Face MoE

```bash
pip install safetensors numpy

python tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/hf-model \
  --out models/model.rabbit \
  --quant 4 --layout square --max-experts 32

./rabbit chat --model models/model.rabbit
```

See [docs/conversion.md](docs/conversion.md).

---

## Architecture (overview)

```
input tokens
    │
    ▼
Attention + KV cache
Dense residual FFN      (RAM)
MoE router + experts    (streamed from disk)
    │
    ▼
lm_head → sample
```

More detail: [docs/architecture.md](docs/architecture.md), [docs/file-format.md](docs/file-format.md).

---

## Project layout

```
src/           Engine (C11)
tools/         Test model builder + HF converter
models/        Local checkpoints
notebooks/     Colab smoke test
docs/          Design notes
```

---

## Limitations

- Synthetic weights do not produce meaningful language; they validate the engine.
- HF conversion uses heuristics for expert tensor names.
- Large frontier MoEs still need aggressive quantization and expert limiting on 4 GB systems.
- Tokenizer is minimal (byte or line-based vocab).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
