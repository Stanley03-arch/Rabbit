# Changelog

## [0.3.0] - 2026-10-25

### Added (Phase B)
- SwiGLU expert path when `expert_dim > hidden_dim` (gate / up / down + SiLU)
- Multi-head attention with RoPE and 256-token KV window
- `rabbit mem --model` memory budget report + suggested max experts on ~4 GB
- Chat startup prints resident estimate

### Compatibility
- Square / Phase A experts (ref-v1, parity) unchanged

## [0.2.5] - 2026-10-25

### Added
- Phase A complete: exact tensor map for OLMoE-family HF MoE (`docs/tensor-map.md`)
- Converter `--arch olmoe` explicit name patterns (non-heuristic)
- Documented packing rule for gate/up/down → square expert records

### Docs
- PHASE_A exit criteria fully checked
- ROADMAP Phase A marked complete

## [0.2.4] - 2026-10-25

### Added
- Phase A ref model, NumPy forward, parity tests
- `rabbit probe` for machine-readable MoE-only fingerprints
- Exact C vs NumPy parity within 1e-4

### Fixed
- Expert GEMV read/write alias (matmul now uses temp delta buffer)

## [0.2.3] - 2026-10-25

- Unit tests, CI, warning cleanup
