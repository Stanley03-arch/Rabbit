# Contributing to Rabbit

Thank you. Rabbit aims for high volume of small, focused, high-quality PRs.

## Setup

```bash
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit
make
make test-model
./rabbit chat --model models/tiny-test.rabbit
```

Python tools:

```bash
pip install numpy safetensors flask
python tools/rabbit_server.py --model models/tiny-test.rabbit
python tools/rabbit_dashboard.py --model models/tiny-test.rabbit
```

Full check before a PR:

```bash
make check
```

## Guidelines

- **C:** C11, clear control flow, low-RAM first. Prefer helpers in `src/utils.c`.
- **Commits:** Small, focused, imperative messages.
- **PRs:** One logical change. Fill the template. Describe what you tested.

## Labels

| Label | Meaning |
|-------|---------|
| `good first issue` | Small, low-risk, ideal first contribution |
| `help wanted` | We want community help |
| `docs` | Documentation |
| `tests` | Tests / CI / parity |
| `bug` | Something broken |
| `enhancement` | New capability |
| `model-support` | New HF name maps / families |
| `windows` | Windows / WSL |

Comment "I'd like to take this" on issues you want to work on.

## High-value areas right now

1. Micro documentation PRs
2. Extra offline stream tests
3. HF expert-name coverage (`tools/rabbit_name_maps.py`)
4. Clearer error messages
5. Bench baselines and structured logs (Phase G)
6. Windows / WSL polish

## License

Contributions are licensed under Apache-2.0.
