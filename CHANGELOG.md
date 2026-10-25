# Changelog

## [0.4.0] - 2026-10-25

### Added (Phase C)
- `rabbit stats --model …` — machine-readable JSON engine metrics (memory, architecture)
- OpenAI-compatible HTTP API: `tools/rabbit_server.py`
  - `POST /v1/chat/completions`
  - `GET /v1/models`, `/v1/stats`, `/health`
- Dashboard now sources model/memory metrics from `rabbit stats` (not log scrape only)
- Dashboard Memory panel (budget / estimate / cache slots)

## [0.3.1] - 2026-10-25

### Added (Phase B continued)
- Memory budget report and hard caps (`RABBIT_MEM_BUDGET_MB`)
- Multi-head attention with per-head RoPE

## [0.3.0] - 2026-10-25

### Added (Phase B start)
- SwiGLU-style expert path when gate/up/down present
- Basic RoPE; flat converter layout ordered for SwiGLU

## [0.2.5] - 2026-10-25

- Phase A closed; tensor map docs; converter heuristics

## [0.2.4] - 2026-10-25

- Ref model, parity tests, probe, GEMV alias fix

## [0.2.3] - 2026-10-25

- Unit tests, CI, warning cleanup
