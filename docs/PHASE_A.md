# Phase A — Correctness spine

Goal: make Rabbit **checkable**, not merely runnable.

Colibrì-level quality is impossible without a correctness spine. This phase defines one frozen toy architecture, a deterministic reference implementation, and automated tests that must stay green.

## Frozen architecture: `rabbit-ref-v1`

| Field | Value |
|-------|-------|
| layers | 2 |
| hidden_dim | 32 |
| vocab_size | 64 (byte subset 0..63) |
| n_experts | 4 |
| experts_per_token | 2 |
| expert matrix | `[hidden_dim x hidden_dim]` int8 with leading `float32` scale |
| dense blob | `hidden_dim * 64` float32 values quantized to int8+scale records for residual |

Routing (reference):

```
score[e] = sum_{i=0..min(31,H-1)} h[i] * ((e+1)*(i+3))
top-2 → softmax weights
```

Expert apply (reference):

```
h := h + 0.05 * w_k * scale * (W_k @ h)   # int8 GEMV, for each selected expert k
```

No attention in the **parity core** (MoE-only). Attention remains experimental outside the ref contract.

## Artifacts

| Path | Role |
|------|------|
| `tools/create_ref_model.py` | Builds `models/ref-v1.rabbit` with **fixed PRNG seed** |
| `tools/ref_forward.py` | NumPy reference forward (MoE-only) |
| `tests/test_parity_ref.py` | Compares C engine fingerprint vs NumPy |
| `tests/test_quant.c` | Quant unit tests |

## Commands

```bash
make ref-model          # write models/ref-v1.rabbit
make test               # C unit tests
make parity             # Python NumPy vs rabbit binary fingerprint
make check              # unit + smoke + parity
```

## Exit criteria for Phase A

- [x] Frozen ref architecture documented
- [x] Deterministic model file generator (seed=1)
- [x] NumPy reference forward
- [x] Automated parity check in CI
- [ ] One **real** small HF MoE with exact tensor map (Phase A.2 — next)

## Non-goals for Phase A

- Matching Colibrì performance or UI
- Full transformer parity (attention/RoPE/norms)
- Frontier model support
