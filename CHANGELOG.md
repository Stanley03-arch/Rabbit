# Changelog

## [0.3.1] - 2026-10-25

### Added (Phase B continued)
- Memory budget report and hard caps at load time
  - Default 1536 MB; override with `RABBIT_MEM_BUDGET_MB`
  - Dense blob that exceeds budget is refused
  - Expert cache slot count chosen from remaining budget
- Multi-head attention with derived `n_heads` / `head_dim` and per-head RoPE

### API
- `rabbit_memory_report()`, `rabbit_print_memory()`, `rabbit_default_mem_budget()`

## [0.3.0] - 2026-10-25

### Added (Phase B start)
- Structured expert apply path: gate / up / down → SwiGLU-style step
- Basic RoPE on the attention path
- Flat converter layout emits gate → up → down order

## [0.2.5] - 2026-10-25

### Added
- Documented tensor map for real HF MoE families
- Improved HF converter expert detection and packing heuristics
- Phase A closed

## [0.2.4] - 2026-10-25

### Added
- Phase A ref model, NumPy forward, parity tests
- `rabbit probe` machine-readable fingerprints
- Exact C vs NumPy parity within 1e-4

### Fixed
- Expert GEMV read/write alias

## [0.2.3] - 2026-10-25

- Unit tests, CI, warning cleanup
