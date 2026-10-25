# Phase D — Quality spine (toward Colibrì-class fidelity)

**Goal:** token-exact agreement with an oracle on a frozen architecture, then lift that path to a real public MoE.

## Milestone M1 — micro oracle (done)

> Rabbit `ref-v2` greedy tokens match the NumPy oracle for ≥ 32 tokens.

| Field | Value |
|-------|-------|
| layers | 2 |
| hidden_dim | 64 |
| n_heads / head_dim | 4 / 16 |
| vocab | 256 (byte) |
| experts / top-k | 4 / 2 |
| intermediate | 128 |
| stack | RMSNorm → RoPE MHA → linear router → SwiGLU |

```bash
make && make parity-v2
./rabbit probe-v2 --model models/ref-v2.rabbit --prompt hi --tokens 32
./rabbit chat --model models/ref-v2.rabbit   # uses faithful forward
```

CI runs `make parity-v2` on every push.

## Milestone M2 — real HF tiny MoE (next)

Target: convert a public micro MoE (e.g. Qwen3-MoE-style tiny checkpoint) into the **same structural layout** the v2 kernels expect:

1. Dense: `embed`, `lm_head`, per-layer `rms_*`, `w_qkv`, `w_o`, `w_router` (float32 or later quant)
2. Experts: `gate` / `up` / `down` int8 (or int4-gs) with exact shapes — **no square crop**
3. Optional: real tokenizer vocab file for byte≠vocab models
4. Parity: greedy tokens vs HuggingFace/transformers for N tokens on a fixed prompt

Until M2 is green, language quality on real checkpoints remains limited.

## Non-goals

- Copying Colibrì engines or containers  
- Claiming frontier quality before M1+M2  
