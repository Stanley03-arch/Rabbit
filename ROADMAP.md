# Roadmap

## Phase A — Correctness spine (complete)

See [docs/PHASE_A.md](docs/PHASE_A.md).

- [x] Frozen `rabbit-ref-v1` architecture
- [x] Deterministic model generator (`make ref-model`)
- [x] NumPy reference forward
- [x] `make parity` + CI hook
- [x] Exact C vs NumPy tensor fingerprint compare (sum / L1 / L2 / first-8)
- [x] One real small HF MoE family with documented tensor map (Qwen3-MoE / Mixtral style)

## Phase B — Engine depth

- Real expert MLP (SwiGLU or family-correct gate/up/down)
- Real attention (RoPE + proper KV)
- Memory budget report + caps
- Full per-layer tensor dump parity (optional deep debug)

## Phase C — Product surface

- OpenAI-compatible HTTP API
- Dashboard metrics from engine (not only log scrape)

## Non-goals

- Claiming Colibrì parity without Phase A+B complete
- Frontier MoEs on 4 GB without severe limits
