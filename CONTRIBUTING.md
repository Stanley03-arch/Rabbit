# Contributing to Rabbit

Thank you for considering a contribution. Rabbit is small on purpose, so a focused change with a
test is the easiest kind to review.

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md). For security
issues, see [SECURITY.md](SECURITY.md) instead of opening a public issue.

## Development setup

```bash
git clone https://github.com/Stanley03-arch/Rabbit.git
cd Rabbit
pip install -r requirements.txt   # only needed for the Python tools and tests
make
make check
```

`make check` is the same command CI runs. It must pass before you open a pull request. Run
`make help` to see every target.

Useful extras while developing:

```bash
make debug                                  # -g -O0 build
make check CFLAGS="-O1 -g -Wall -Wextra -std=c11 -I./src -fsanitize=address,undefined" \
           LDFLAGS="-lm -fsanitize=address,undefined"   # what the sanitizer CI job does
```

## How correctness is checked

The C engine is validated against an independent NumPy implementation (`tools/ref_forward*.py`).
The tests build a small model, run both implementations on the same weights and prompts, and
require identical greedy token output. If you change numerics (quantization, attention, MoE
routing, file layout), the matching NumPy oracle usually needs the same change, and that is the
test that tells you whether you got it right.

| Change touches | Test that covers it |
|----------------|---------------------|
| Quantization kernels, tokenizer | `make test` |
| v1 dense/MoE path | `make parity` |
| v2 forward pass (attention, MoE, final norm) | `make parity-v2` |
| Converter, HF tensor naming | `make m2`, `make mixtral-micro` |
| Raw f32 experts | `make exact-f32` |
| CLI end to end | `make smoke` |

## Code guidelines

- **C:** C11, no dependencies beyond libc and libm. The build uses `-Wall -Wextra` and CI adds
  `-Werror`, so keep it warning-free. Prefer clear control flow over cleverness.
- **Memory:** Assume low-RAM targets. Avoid unbounded allocations in the hot path.
- **Format:** `.editorconfig` and `.clang-format` describe the style (4 spaces). Match neighbouring
  code rather than reformatting files wholesale.
- **Python:** keep tools dependency-light (`numpy` for tests; `safetensors` and `flask` only where
  needed).
- **Commits:** small, focused, imperative messages
  (`Add int4 GEMV path`, not `updates`).

## What is most useful right now

1. Correctness tests for expert fetch and quantization kernels
2. Better Hugging Face tensor-name coverage for specific model families
3. Real-model end-to-end parity against a transformers reference (see [ROADMAP.md](ROADMAP.md))
4. Documentation fixes and clearer error messages
5. Tokenizer backends (SentencePiece / HF tokenizers) behind the existing API

## Pull requests

1. Fork and branch from `main`.
2. Keep changes scoped, and add or update a test.
3. Update docs when behavior, CLI flags or the file format change, and add a line to
   [CHANGELOG.md](CHANGELOG.md).
4. Fill in the pull request template, including what you ran.

## Reporting issues

Use the bug report form. The most useful reports include the output of `./rabbit version`, OS and
compiler, the exact command line, the model source (synthetic vs converted, and converter flags),
and relevant log output.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0.
