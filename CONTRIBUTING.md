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
./rabbit stats --model models/tiny-test.rabbit
./rabbit bench --model models/tiny-test.rabbit --tokens 8 --runs 2
```

Python tooling (converter + UIs):

```bash
pip install numpy safetensors flask   # + gradio for tools/rabbit_ui.py
python tools/convert_hf_to_rabbit.py --demo --out models/demo-moe.rabbit
python tools/rabbit_server.py --model models/tiny-test.rabbit
python tools/rabbit_dashboard.py --model models/tiny-test.rabbit
```

Full local check (recommended before a PR):

```bash
make check   # includes unit tests, smoke, bench-smoke, parity, m2, mixtral
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

## CLI surface (keep in sync)

| Command | Role |
|---------|------|
| `chat` | Interactive generation |
| `stats` | JSON model / memory report |
| `bench` | Timed ref-v2 (or probe fallback); `--runs` |
| `probe` / `probe-v2` | One-shot hidden / token dumps |
| `version` | Print version string |

Optional flags used across tools: `--model`, `--tokenizer`, `--tokens`, `--temp`, `--prompt`, `--runs`.

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
