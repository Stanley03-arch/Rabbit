# Changelog

## [0.2.4] - 2026-10-25

### Added
- Phase A ref model, NumPy forward, parity tests
- `rabbit probe` for machine-readable MoE-only fingerprints
- Exact C vs NumPy parity within 1e-4

### Fixed
- Expert GEMV read/write alias (matmul now uses temp delta buffer)

## [0.2.3] - 2026-10-25

- Unit tests, CI, warning cleanup
