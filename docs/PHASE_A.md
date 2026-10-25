# Phase A — Correctness spine

**Status: complete (v0.2.5)**

Goal: make Rabbit **checkable**, not merely runnable.

## Frozen architecture: `rabbit-ref-v1`

| Field | Value |
|-------|-------|
| layers | 2 |
| hidden_dim | 32 |
| vocab_size | 64 |
| n_experts | 4 |
| experts_per_token | 2 |
| expert | int8 `H x H` + leading float32 scale |

Routing and apply match `tools/ref_forward.py` and `rabbit_ref_forward` in C.

## Commands

```bash
make ref-model
make
./rabbit probe --model models/ref-v1.rabbit --prompt hi
make parity          # exact C vs NumPy (1e-4)
make check
```

## Exit criteria

- [x] Frozen ref architecture documented
- [x] Deterministic model file generator (seed=1)
- [x] NumPy reference forward
- [x] `rabbit probe` machine-readable fingerprint
- [x] Automated **exact** C vs NumPy parity (`make parity`)
- [x] Expert GEMV uses temp buffer (no read/write alias)
- [x] Real HF MoE family tensor map documented (see [tensor-map.md](tensor-map.md))

## Non-goals

- Colibrì performance/UI parity
- Full transformer (attention) inside the ref contract
