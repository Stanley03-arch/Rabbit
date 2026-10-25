# Tensor map — HF MoE → Rabbit

## Layout modes

| `--layout` | Dense | Experts | Engine path |
|------------|-------|---------|-------------|
| `square` (legacy) | opaque packed | single H×H (lossy crop) | legacy |
| `flat` | opaque packed | gate→up→down ordered | SwiGLU pieces |
| **`v2` (M2)** | **structured float32** | **exact gate/up/down, per-layer** | **`ref-v2` faithful** |

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
Per expert record:
  gate_proj  [I, H]  int8+scale
  up_proj    [I, H]  int8+scale
  down_proj  [H, I]  int8+scale

Disk order (per-layer):
  L0E0, L0E1, … L0E{E-1}, L1E0, … L{L-1}E{E-1}

Header:
  n_experts           = E  (experts per layer / router width)
  total_experts_size  = L × E × expert_size

C lookup:
  global_id = layer * n_experts + local_id   (unless experts_shared)
  rabbit_expert_ptr_layer(m, layer, e)
```

Recognized HF names: Qwen3-MoE / Mixtral / DeepSeek-style `layers.L.mlp.experts.E.(gate|up|down)_proj`.

Legacy models with `total_experts_size / expert_size == n_experts` stay in **shared** mode.

## Micro fixture (CI, no download)

```bash
make micro-hf    # models/hf_micro_moe + models/micro-hf-v2.rabbit
make m2          # convert + C↔NumPy parity on the converted file
```

Fixture and `create_ref_v2` emit **distinct experts per layer**.

## Real checkpoint

```bash
pip install safetensors numpy
python tools/convert_hf_to_rabbit.py \
  --model-dir /path/to/hf \
  --out models/real-v2.rabbit \
  --layout v2 --quant 8
./rabbit chat --model models/real-v2.rabbit
```

Large models still need aggressive expert limits / quant; v2 is about **shape fidelity**, not RAM magic.

See [PHASE_D.md](PHASE_D.md).
