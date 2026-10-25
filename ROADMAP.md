# Roadmap

## Phase A — Correctness spine (complete)

See [docs/PHASE_A.md](docs/PHASE_A.md).

## Phase B — Engine depth (mostly complete)

Memory + SwiGLU + multi-head RoPE done.

## Phase C — Product surface (mostly complete)

OpenAI API + SSE, `rabbit stats`, dashboard engine metrics.

## Phase D — Quality spine

See [docs/PHASE_D.md](docs/PHASE_D.md).

### M1 — micro oracle
- [x] `rabbit-ref-v2` + NumPy oracle + C `forward_v2.c`
- [x] `make parity-v2` + CI
- [x] Chat uses ref-v2 path for matching models

### M2 — HF-shaped conversion
- [x] `--layout v2` exact dense + gate/up/down
- [x] Deterministic HF-like micro fixture (`make micro-hf`)
- [x] `make m2` / CI + per-layer experts + f32 path
- [x] Real BPE tokenizer (RBTTOK01)
- [x] Group-scaled int4 (group=32)
- [x] GQA dense layout (Q/K/V separate, header.reserved metadata)
- [ ] Live public MoE end-to-end (TinyBrainBot 360M Mixtral-style) — convert + smoke

## Non-goals

- Claiming Colibrì parity without real-model transformers parity
- Copying Colibrì engines, UI, or containers
