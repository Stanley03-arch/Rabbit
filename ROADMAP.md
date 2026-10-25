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

## Phase C — Product surface (mostly complete)

- [x] OpenAI-compatible HTTP API (`tools/rabbit_server.py`)
- [x] Engine metrics command (`rabbit stats` JSON)
- [x] Dashboard metrics from engine (not only log scrape)
- [x] Streaming completions (SSE)
- [ ] Auth / rate limits (optional)

## Phase D — Quality spine (toward Colibrì-class fidelity)

See [docs/PHASE_D.md](docs/PHASE_D.md).

- [x] Frozen `rabbit-ref-v2` architecture (RMSNorm, RoPE, SwiGLU, real router)
- [x] Deterministic generator (`make ref-v2`)
- [x] NumPy full forward oracle (`tools/ref_forward_v2.py`)
- [x] Oracle self-check (`make parity-v2`)
- [ ] C forward matches NumPy argmax for ≥32 greedy tokens
- [ ] Wire `parity-v2` into CI once C match is green
- [ ] Real HF tiny MoE with exact tensor map (no square crop)

## Non-goals

- Claiming Colibrì parity without Phase D M1 (token-exact micro oracle) complete
- Frontier MoEs on 4 GB without severe limits
- Copying Colibrì engines, UI, or containers
