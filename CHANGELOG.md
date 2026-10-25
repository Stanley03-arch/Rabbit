# Changelog

All notable changes are listed here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.4.8] - 2026-10-25

### Fixed
- **v2 forward pass treated raw-f32 experts as int8.** `expert_quant == 0` means f32 in the
  file format, but `forward_v2.c` defaulted `0` to `8`, so the f32 branch was unreachable and
  f32 models produced wrong logits. `make parity-v2` and `make exact-f32` were failing on `main`
  because of this; both pass again.
- The binary reported `0.4.6` while the changelog was at `0.4.7`; the version string is now
  in sync.

### Changed
- **`rabbit_server.py` and `rabbit_dashboard.py` now bind to `127.0.0.1` by default** instead of
  `0.0.0.0`. Neither has authentication, so exposing them on every interface was an unsafe
  default. Pass `--host 0.0.0.0` to restore the old behavior (the Colab notebook already does).
- CI now runs the whole `make check` (it previously skipped `exact-f32` and `mixtral-micro`,
  which is how the bug above went unnoticed), builds with `-Werror`, covers gcc and clang on
  Linux plus clang on macOS, and adds an ASan + UBSan job.
- README rewritten with a quick start, architecture overview, verification table and docs index.

### Added
- `make help`.
- Issue forms (bug, feature), pull request template, `SECURITY.md`, `CODE_OF_CONDUCT.md`,
  Dependabot for GitHub Actions, `.editorconfig`, `.clang-format`, `requirements.txt`.
- Project logo in `assets/`.

## [0.4.7] - 2026-10-25

### Final RMSNorm
- Dense layout: `embed | lm_head | final_norm | layers…`
- C `forward_v2` + NumPy oracle apply `model.norm` before logits
- Fixtures / converter pack `model.norm.weight`

## [0.4.6] - 2026-10-25

### GQA + Mixtral path
- Separate Q/K/V, header GQA metadata, Mixtral w1/w3/w2, `make mixtral-micro`

## [0.4.5] - 2026-10-25

### Group-scaled int4
- group=32 pack/dequant/GEMV + converter `--quant 4`

## [0.4.4] - 2026-10-25

### Real BPE tokenizer (RBTTOK01)

## [0.4.3] — exact f32 experts

## [0.4.2] — per-layer expert tables

## [0.4.1] — Phase D M1/M2 start

## [0.4.0] — OpenAI API + SSE
