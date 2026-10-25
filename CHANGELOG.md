# Changelog

## [0.2.5] - 2026-10-25

### Added
- Documented tensor map for real HF MoE families (Qwen3-MoE / Mixtral style)
- Improved HF converter expert detection and packing heuristics
- Explicit notes for converting tiny public MoE models (e.g. micro Qwen3-MoE)
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
