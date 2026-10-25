# Roadmap

## Phase A — Correctness spine (in progress)

See [docs/PHASE_A.md](docs/PHASE_A.md).

- [x] Frozen `rabbit-ref-v1` architecture
- [x] Deterministic model generator (`make ref-model`)
- [x] NumPy reference forward
- [x] `make parity` + CI hook
- [ ] Exact C vs NumPy tensor dump compare
- [ ] One real small HF MoE with documented tensor map

## Phase B — Engine depth

- Real expert MLP (SwiGLU or family-correct)
- Real attention (RoPE + proper KV)
- Memory budget report + caps

## Phase C — Product surface

- OpenAI-compatible HTTP API
- Dashboard metrics from engine (not only log scrape)

## Non-goals

- Claiming Colibrì parity without Phase A+B complete
- Frontier MoEs on 4 GB without severe limits
