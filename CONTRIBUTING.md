# Contributing to Rabbit

Thank you for considering a contribution.

## Development setup

```bash
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit
make
make test-model
./rabbit chat --model models/tiny-test.rabbit
./rabbit version
```

Python tooling (converter):

```bash
pip install numpy safetensors
python tools/convert_hf_to_rabbit.py --demo --out models/demo-moe.rabbit
```

Full local check (recommended before a PR):

```bash
make check
```

## Code guidelines

- **C:** C11, no unnecessary dependencies. Prefer clear control flow over cleverness.
- **Memory:** Assume low-RAM targets. Avoid unbounded allocations in the hot path.
  Prefer the helpers in `src/utils.c`:
  - `rabbit_xmalloc` / `rabbit_xcalloc` / `rabbit_xrealloc` / `rabbit_xstrdup`
  - `rabbit_die` / `rabbit_dief` for fatal errors
- **Format:** Match existing style (4-space or as in neighboring files; keep headers consistent with `rabbit.h`).
- **Commits:** Prefer small, focused commits with imperative messages  
  (e.g. `Add int4 GEMV path`, not `updates`).

## What is most useful right now

1. Correctness tests for expert fetch / quant kernels
2. Better HF expert-name coverage for specific model families
3. Documentation fixes and clearer error messages
4. Real tokenizer backends (SentencePiece / HF tokenizers) behind the existing API
5. Live public MoE end-to-end validation (`make live-moe`)

## Pull requests

1. Fork and branch from `main`
2. Keep changes scoped
3. Update docs / CHANGELOG when behavior or CLI changes
4. Describe what you tested (`make`, `make test-model`, `make check`, converter flags, etc.)

## Reporting issues

Include:

- OS and compiler version
- Exact command line
- Model source (synthetic vs converted)
- Relevant log output

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
