# Phase A — Correctness spine

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
- [x] One real small HF MoE with exact tensor map

The real-map item is documented in [tensor-map.md](tensor-map.md) (OLMoE family)
and implemented via `tools/convert_hf_to_rabbit.py --arch olmoe`.

## Non-goals

- Colibrì performance/UI parity
- Full transformer (attention) inside the ref contract
- Bit-exact HF language quality for OLMoE (Phase B)
