# Roadmap

## Phase A — Correctness spine (complete)

See [docs/PHASE_A.md](docs/PHASE_A.md) and [docs/tensor-map.md](docs/tensor-map.md).

- [x] Frozen `rabbit-ref-v1` architecture
- [x] Deterministic model generator (`make ref-model`)
- [x] NumPy reference forward
- [x] `make parity` + CI hook
- [x] Exact C vs NumPy tensor dump compare (via probe fingerprints)
- [x] One real small HF MoE with documented tensor map (OLMoE)

## Phase B — Engine depth (in progress)

See [docs/PHASE_B.md](docs/PHASE_B.md).

- [x] Real expert MLP (SwiGLU) layout + kernel with square fallback
- [x] RoPE + multi-head KV attention (sliding window)
- [x] Memory budget report (`rabbit mem`) + 4 GB suggestion
- [ ] Converter `--layout swiglu` packs gate/up/down without square collapse
- [ ] Teacher-forcing parity against a tiny HF MoE subset (optional)

## Phase C — Product surface

- OpenAI-compatible HTTP API
- Dashboard metrics from engine (not only log scrape)

## Non-goals

- Claiming Colibrì parity without Phase A+B complete
- Frontier MoEs on 4 GB without severe limits
