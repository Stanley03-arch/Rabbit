# Contributing to Rabbit

Thank you for considering a contribution. Rabbit aims to be the easiest high-quality MoE inference engine for constrained hardware. We want **lots** of small, focused PRs.

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
make check   # unit tests, smoke, bench-smoke, parity, m2, mixtral, ...
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

## Labels we use

| Label | Meaning |
|-------|---------|
| `good first issue` | Small, well-scoped, low-risk. Ideal first contribution |
| `help wanted` | We would love community help |
| `docs` | Documentation only |
| `tests` | Tests / CI / parity |
| `bug` | Something is broken |
| `enhancement` | New capability |
| `model-support` | New HF name maps / architecture family |
| `performance` | Speed or memory |
| `windows` | Windows / WSL specific |

Look for open issues with these labels. Comment "I'd like to take this" so we can assign you.

## What is most useful right now

1. **Micro documentation PRs** — clarify a section of README, ROADMAP, docs/*.md, or add a usage example
2. **More offline stream tests** for additional MoE families
3. **HF expert-name coverage** for new model families (see `tools/rabbit_name_maps.py`)
4. **Better error messages** in the converter and CLI
5. **Bench baseline tables** and structured request logging (see ROADMAP Phase G)
6. **Windows / WSL polish**
7. Real tokenizer backends behind the existing API (SentencePiece / HF)

## Pull requests

1. Fork and branch from `main`
2. Keep changes scoped (one logical change per PR)
3. Update docs / CHANGELOG when behavior or CLI changes
4. Describe what you tested (`make`, `make test-model`, `make check`, converter flags, etc.)
5. Fill in the PR template

We merge small, correct PRs quickly. Large design changes should start as an issue first.

## Reporting issues

Include:

- OS and compiler version
- Exact command line
- Model source (synthetic vs converted)
- Relevant log output

Use the issue templates — they make triage much faster.

## Recognition

Every meaningful contribution is valued. We plan to adopt an all-contributors style recognition once volume grows.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
