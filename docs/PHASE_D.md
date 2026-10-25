# Phase D — Quality spine (toward Colibrì-class fidelity)

**Goal:** one checkable forward path that produces **token-exact** agreement with a NumPy/PyTorch oracle on a frozen tiny MoE — the same *kind* of bar Colibrì uses (oracle parity), at micro scale.

This is **not** a claim of Colibrì product parity. It is the prerequisite for real language quality.

## Milestone M1 (current target)

> Rabbit `ref-v2` greedy tokens match the NumPy oracle for ≥ 32 tokens on a fixed prompt set.

### Frozen architecture: `rabbit-ref-v2`

| Field | Value |
|-------|-------|
| layers | 2 |
| hidden_dim | 64 |
| n_heads | 4 |
| head_dim | 16 |
| vocab_size | 256 |
| n_experts | 4 |
| experts_per_token | 2 |
| expert intermediate | 128 |
| expert | SwiGLU: gate, up (`I×H`), down (`H×I`), int8 + scale |
| norm | RMSNorm (eps=1e-5) |
| rope | standard, theta=10000 |
| router | linear `E×H` then softmax top-k |

Dense blob (ordered):

1. `embed` — `vocab × hidden` float32  
2. `lm_head` — `hidden × vocab` float32 (untied for the ref)  
3. Per layer: `rms_attn`, `rms_ffn` (hidden each), `w_qkv` (`3H×H`), `w_o` (`H×H`), `w_router` (`E×H`)

Experts: flat layout, gate → up → down, each `[scale][int8 payload]`.

### Commands

```bash
make && make ref-v2
make parity-v2
./rabbit probe-v2 --model models/ref-v2.rabbit --prompt hi --tokens 32
python3 tools/ref_forward_v2.py --model models/ref-v2.rabbit --prompt hi --tokens 32
```

### Exit criteria for M1

- [x] Architecture frozen and documented  
- [x] Deterministic generator (`seed=2`)  
- [x] NumPy full forward (embed → layers → logits)  
- [x] C forward (`src/forward_v2.c`) + `probe-v2`  
- [x] `make parity-v2` harness (C vs NumPy)  
- [ ] Verify green on your machine / CI  

## Later (M2+)

- Real HF tiny MoE (exact tensor map, no square crop)  
- Group-scaled int4  
- Real SentencePiece / BPE tokenizer  
- SIMD / prefetch (speed only after correctness)

## Non-goals

- Copying Colibrì engines, UI, or containers  
- Claiming frontier-model quality before M1 is green  
