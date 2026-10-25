# Rabbit

**Local MoE inference for constrained hardware.**

Rabbit is a small pure-C engine that runs Mixture-of-Experts models by keeping dense weights in RAM and streaming routed experts from disk. The design target is machines with limited memory (including ~4 GB class laptops).

Inspired by [Colibrì](https://github.com/JustVugg/colibri).

> **Status:** early development (v0.2.1). The full pipeline works on synthetic models. Real open MoE conversion is supported but still evolving.

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
- POSIX (`mmap`, `pread`-style I/O)
- Optional: Python 3 + `numpy` + `safetensors` for model conversion

---

## Build

```bash
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit
make
```

Artifacts:

- `./rabbit` — inference binary
- `make test-model` — tiny synthetic checkpoint
- `make demo-model` — larger int4 demo MoE (requires Python 3)

---

## Quick start

```bash
make test-model
./rabbit chat --model models/tiny-test.rabbit

make demo-model
./rabbit chat --model models/demo-moe.rabbit --tokens 64 --temp 0.8
```

Optional word vocabulary:

```bash
./rabbit chat --model models/demo-moe.rabbit --vocab models/sample_vocab.txt
```

---

## CLI

```
rabbit version
rabbit chat --model <path> [options]

Options:
  --model <path>   Path to a .rabbit model (required)
  --vocab <path>   Word vocabulary file (one token per line)
  --tokens <n>     Maximum new tokens to generate (default: 64)
  --temp <f>       Sampling temperature (default: 0.85)
  -h, --help       Show help
```

---

## Convert a Hugging Face MoE

```bash
pip install safetensors numpy

python tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/hf-model \
  --out models/model.rabbit \
  --quant 4 \
  --layout square \
  --max-experts 32

./rabbit chat --model models/model.rabbit
```

See [docs/conversion.md](docs/conversion.md) for layout and quantization details.

---

## Architecture (overview)

```
input tokens
    │
    ▼
┌──────────────────┐
│  Attention + KV cache  │  (short context, grouped)
├──────────────────┤
│  Dense residual FFN    │  (always in RAM)
├──────────────────┤
│  MoE router + experts  │  (streamed from disk)
└──────────────────┘
    │
    ▼
lm_head → sample
```

Experts are stored in a single file region and mapped with `mmap`. A small LRU keeps hot experts in RAM. See [docs/architecture.md](docs/architecture.md) and [docs/file-format.md](docs/file-format.md).

---

## Project layout

```
src/           Engine (C11)
tools/         Test model builder + HF converter
models/        Local checkpoints (not tracked except samples)
docs/          Design and conversion notes
```

---

## Limitations

- Generation quality on synthetic weights is not meaningful language; it validates the pipeline.
- Hugging Face conversion uses heuristics for expert tensor names; not every architecture is perfect yet.
- Full frontier MoE models (e.g. multi-hundred-GB checkpoints) still require aggressive quantization and expert limiting on 4 GB systems.
- Tokenizer is intentionally minimal (byte or line-based vocab), not SentencePiece/tiktoken yet.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports and focused pull requests are welcome.

---

## License

Apache License 2.0. See [LICENSE](LICENSE).
