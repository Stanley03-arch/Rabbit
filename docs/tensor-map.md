# Tensor map — HF MoE → Rabbit

## Layout modes

| `--layout` | Dense | Experts | Engine path |
|------------|-------|---------|-------------|
| `square` (legacy) | opaque packed | single H×H (lossy crop) | legacy |
| `flat` | opaque packed | gate→up→down ordered | SwiGLU pieces |
| **`v2` (M2)** | **structured float32** | **exact gate/up/down shapes** | **`ref-v2` faithful** |

## v2 dense layout (exact)

```
embed          [V, H] f32
lm_head        [H, V] f32     (transposed if HF stores [V, H])
per layer L:
  rms_attn     [H]            ← input_layernorm
  rms_ffn      [H]            ← post_attention_layernorm
  w_qkv        [3H, H]        ← stack(q_proj, k_proj, v_proj)
  w_o          [H, H]         ← o_proj
  w_router     [E, H]         ← mlp.gate / router
```

## v2 expert layout (exact — no square crop)

```
gate_proj  [I, H]  int8+scale
up_proj    [I, H]  int8+scale
down_proj  [H, I]  int8+scale
```

Recognized HF names: Qwen3-MoE / Mixtral / DeepSeek-style `experts.N.(gate|up|down)_proj`.

## Micro fixture (CI, no download)

```bash
make micro-hf    # models/hf_micro_moe + models/micro-hf-v2.rabbit
make m2          # convert + C↔NumPy parity on the converted file
```

Experts in the fixture are **shared** (layer-0 only) to match the current global expert table in `forward_v2.c`.

## Real checkpoint

```bash
pip install safetensors numpy
# after downloading a tiny MoE to /path/to/hf
python tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/hf \
  --out models/real-v2.rabbit \
  --layout v2 --quant 8
./rabbit chat --model models/real-v2.rabbit
```

Large models still need aggressive expert limits / quant; v2 is about **shape fidelity**, not RAM magic.

See [PHASE_D.md](PHASE_D.md) and [conversion.md](conversion.md).
