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

## Phase D — Quality spine (M1 largely done)

See [docs/PHASE_D.md](docs/PHASE_D.md).

- [x] Frozen `rabbit-ref-v2` architecture
- [x] NumPy oracle + C faithful forward (`src/forward_v2.c`)
- [x] `rabbit probe-v2` + `make parity-v2`
- [x] CI runs `parity-v2`
- [x] Chat auto-selects ref-v2 forward for matching models
- [ ] M2: real HF tiny MoE with exact tensor map into ref-v2 layout
- [ ] Group-scaled int4 + real BPE tokenizer

## Non-goals

- Claiming Colibrì parity without verified M1 + real-model M2
- Frontier MoEs on 4 GB without severe limits
- Copying Colibrì engines, UI, or containers
