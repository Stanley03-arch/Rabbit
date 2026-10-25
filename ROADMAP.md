# Roadmap

## Phase A — Correctness spine (complete)

See [docs/PHASE_A.md](docs/PHASE_A.md) and [docs/tensor-map.md](docs/tensor-map.md).

- [x] Frozen `rabbit-ref-v1` architecture
- [x] Deterministic model generator (`make ref-model`)
- [x] NumPy reference forward
- [x] `make parity` + CI hook
- [x] Exact C vs NumPy tensor dump compare (via probe fingerprints)
- [x] One real small HF MoE with documented tensor map (OLMoE)

## Phase B — Engine depth

- Real expert MLP (SwiGLU or family-correct) — replace square collapse
- Real attention (RoPE + proper KV)
- Memory budget report + caps
- Optional: teacher-forcing parity against a tiny HF MoE subset

## Phase C — Product surface

- OpenAI-compatible HTTP API
- Dashboard metrics from engine (not only log scrape)

## Non-goals

- Claiming Colibrì parity without Phase A+B complete
- Frontier MoEs on 4 GB without severe limits
