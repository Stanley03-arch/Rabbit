# Changelog

## [0.4.1] - 2026-10-25

### Added (Phase D — C side)
- `src/forward_v2.c` — faithful ref-v2 forward (RMSNorm, RoPE, multi-head attn, linear router, SwiGLU, lm_head)
- `rabbit probe-v2 --model … --prompt … --tokens N` — greedy token dump for parity
- `make parity-v2` compares C tokens to the NumPy oracle

## [0.4.0] - 2026-10-25

### Added (Phase C)
- `rabbit stats`, OpenAI-compatible API + SSE, dashboard engine metrics

### Added (Phase D start)
- `rabbit-ref-v2` generator + NumPy oracle + design doc

## [0.3.1] - 2026-10-25

- Memory budget + multi-head RoPE attention

## [0.3.0] - 2026-10-25

- SwiGLU expert path; basic RoPE

## [0.2.5] - 2026-10-25

- Phase A closed; tensor map docs

## [0.2.4] - 2026-10-25

- Ref model, parity tests, probe

## [0.2.3] - 2026-10-25

- Unit tests, CI, warning cleanup
