# Contributing to Rabbit

Thank you for considering a contribution.

## Development setup

```bash
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit
make
make test-model
./rabbit chat --model models/tiny-test.rabbit
```

Python tooling (converter):

```bash
pip install numpy safetensors
python tools/convert_hf_to_rabbit.py --demo --out models/demo-moe.rabbit
```

## Code guidelines

- **C:** C11, no unnecessary dependencies. Prefer clear control flow over cleverness.
- **Memory:** Assume low-RAM targets. Avoid unbounded allocations in the hot path.
- **Format:** Match existing style (4-space or as in neighboring files; keep headers consistent with `rabbit.h`).
- **Commits:** Prefer small, focused commits with imperative messages  
  (e.g. `Add int4 GEMV path`, not `updates`).

## What is most useful right now

1. Correctness tests for expert fetch / quant kernels
2. Better HF expert-name coverage for specific model families
3. Documentation fixes and clearer error messages
4. Real tokenizer backends (SentencePiece / HF tokenizers) behind the existing API

## Pull requests

1. Fork and branch from `main`
2. Keep changes scoped
3. Update docs when behavior or CLI changes
4. Describe what you tested (`make`, `make test-model`, converter flags, etc.)

## Reporting issues

Include:

- OS and compiler version
- Exact command line
- Model source (synthetic vs converted)
- Relevant log output

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
