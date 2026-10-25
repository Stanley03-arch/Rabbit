# Changelog

## [0.4.1] - 2026-10-25

### Added (Phase D — C side)
- `src/forward_v2.c` — faithful ref-v2 forward
- `rabbit probe-v2` + `make parity-v2` (C vs NumPy)
- CI runs `parity-v2`
- Chat auto-uses ref-v2 kernels when model matches (`ref-v2` / float32 dense layout)

## [0.4.0] - 2026-10-25

### Added (Phase C)
- `rabbit stats`, OpenAI API + SSE, dashboard engine metrics

### Added (Phase D start)
- `rabbit-ref-v2` generator + NumPy oracle

## [0.3.1] - 2026-10-25

- Memory budget + multi-head RoPE attention

## [0.3.0] - 2026-10-25

- SwiGLU expert path; basic RoPE

## [0.2.5] – [0.2.3] - 2026-10-25

- Phase A parity, tests, CI
