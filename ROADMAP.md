# Roadmap

## Phase A — Correctness spine (complete)

See [docs/PHASE_A.md](docs/PHASE_A.md).

- [x] Frozen `rabbit-ref-v1` architecture
- [x] Deterministic model generator (`make ref-model`)
- [x] NumPy reference forward
- [x] `make parity` + CI hook
- [x] Exact C vs NumPy tensor fingerprint compare
- [x] One real small HF MoE family with documented tensor map

## Phase B — Engine depth (mostly complete)

- [x] Real expert MLP path (SwiGLU-style when gate/up/down data present)
- [x] Basic RoPE in attention
- [x] Multi-head attention with derived head shapes + per-head RoPE
- [x] Memory budget report + hard caps (`RABBIT_MEM_BUDGET_MB`)
- [ ] Full per-layer tensor dump parity (optional deep debug)

## Phase C — Product surface

- OpenAI-compatible HTTP API
- Dashboard metrics from engine (not only log scrape)

## Non-goals

- Claiming Colibrì parity without Phase A+B complete
- Frontier MoEs on 4 GB without severe limits
