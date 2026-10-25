# Phase B — Engine depth

Goal: replace synthetic kernels with family-correct structure so real MoE weights can run without square collapse.

## Scope

| Item | Status |
|------|--------|
| Real expert MLP (SwiGLU) | In progress — layout + kernel |
| RoPE + multi-head KV attention | In progress |
| Memory budget report + caps | In progress |
| Teacher-forcing parity on tiny HF subset | Optional / later |

## Expert record v2 (SwiGLU)

When `expert_dim > hidden_dim` (or converter `--layout swiglu`), each expert is:

```
float32 scale_gate
packed gate_proj   # int8/int4, shape [intermediate, hidden] row-major
float32 scale_up
packed up_proj     # [intermediate, hidden]
float32 scale_down
packed down_proj   # [hidden, intermediate]
```

Apply:

```
gate = silu(scale_gate * (W_gate @ x))
up   = scale_up * (W_up @ x)
h   += scale_down * (W_down @ (gate * up))
```

Legacy square experts (`expert_dim == hidden_dim` or payload too small) keep the Phase A path so `make parity` still passes.

## Attention

- Multi-head (groups from `hidden_dim` / head size, default head_dim=32 or 64)
- Rotary position embeddings (RoPE) on Q and K
- Sliding KV window (default 256)
- Residual mix after attention

## Memory budget

On load / `rabbit mem --model …`:

- Dense resident bytes
- Expert record size × cache slots
- KV cache estimate
- Suggested max experts for a RAM target

## Commands

```bash
make && make check
./rabbit mem --model models/ref-v1.rabbit
./rabbit chat --model models/tiny-test.rabbit
```

Convert OLMoE without square collapse:

```bash
python3 tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/OLMoE \
  --out models/olmoe.rabbit \
  --arch olmoe --layout swiglu --quant 4 --max-experts 16
```
