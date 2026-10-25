# Changelog

## [0.3.0] - 2026-10-25

### Added (Phase B start)
- Structured expert apply path: when the expert blob is large enough it is treated as gate / up / down and a SwiGLU-style step is performed
- Basic RoPE on the attention query/key path
- Flat converter layout now emits tensors in gate → up → down order for the new kernel

### Notes
- Classic square-expert path (used by the ref model and parity tests) is unchanged and still exact
- Full production-grade SwiGLU + proper multi-head RoPE still evolving

## [0.2.5] - 2026-10-25

### Added
- Documented tensor map for real HF MoE families (Qwen3-MoE / Mixtral style)
- Improved HF converter expert detection and packing heuristics
- Explicit notes for converting tiny public MoE models
- ROADMAP / PHASE_A updated to close remaining Phase A items

### Changed
- Version bump to 0.2.5

## [0.2.4] - 2026-10-25

### Added
- Phase A ref model, NumPy forward, parity tests
- `rabbit probe` for machine-readable MoE-only fingerprints
- Exact C vs NumPy parity within 1e-4

### Fixed
- Expert GEMV read/write alias (matmul now uses temp delta buffer)

## [0.2.3] - 2026-10-25

- Unit tests, CI, warning cleanup
