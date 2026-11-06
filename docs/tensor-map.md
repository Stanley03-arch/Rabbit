# Tensor map — HF MoE → Rabbit

## Layout modes

| `--layout` | Dense | Experts | Engine path |
|------------|-------|---------|-------------|
| `square` (legacy) | opaque packed | single H×H (lossy crop) | legacy |
| `flat` | opaque packed | gate→up→down ordered | SwiGLU pieces |
| **`v2` (M2)** | **structured float32** | **exact gate/up/down, per-layer** | **`ref-v2` faithful** |

Streaming converter (`convert_hf_stream.py`) always writes **v2-shaped** files.

## v2 dense layout (exact)

```
embed          [V, H] f32
lm_head        [H, V] f32     (transposed if HF stores [V, H])
final_norm     [H]            ← model.norm
per layer L:
  rms_attn     [H]            ← input_layernorm / attention_norm
  rms_ffn      [H]            ← post_attention_layernorm / ffn_norm
  w_q          [n_heads*hd, H]
  w_k          [n_kv*hd, H]
  w_v          [n_kv*hd, H]
  w_o          [H, n_heads*hd]
  w_router     [E, H]         ← block_sparse_moe.gate / mlp.gate / router
```

## v2 expert layout (exact — no square crop)

```
Per expert record:
  gate  [I, H]   f32 or group-int4
  up    [I, H]   f32 or group-int4
  down  [H, I]   f32 or group-int4

Disk order (per-layer):
  L0E0, L0E1, … L0E{E-1}, L1E0, … L{L-1}E{E-1}

Header:
  n_experts           = E  (experts per layer / router width)
  total_experts_size  = L × E × expert_size

C lookup:
  global_id = layer * n_experts + local_id   (unless experts_shared)
  rabbit_expert_ptr_layer(m, layer, e)
```

## Recognized HF names

Canonical table lives in `tools/rabbit_name_maps.py`.

| Family | Gate / up / down | Router |
|--------|------------------|--------|
| **Mixtral** | `…block_sparse_moe.experts.{e}.w1` / `w3` / `w2` | `…block_sparse_moe.gate` |
| **Qwen2/3-MoE** | `…mlp.experts.{e}.gate_proj` / `up_proj` / `down_proj` | `…mlp.gate` |
| **moe.*** | `…moe.experts.{e}.gate_proj|w1` / … | `…moe.gate` or `…moe.router` |
| **sparse_moe** | `…sparse_moe.experts.{e}.w1\|w3\|w2` | `…sparse_moe.gate` |

Also scanned as expert markers: `.experts.`, `mlp.experts`, `moe.experts`, `feed_forward.experts`.

Config expert count keys: `num_local_experts`, `num_experts`, `n_routed_experts`.

Legacy models with `total_experts_size / expert_size == n_experts` stay in **shared** mode.

## Micro fixtures (CI, no download)

```bash
make micro-hf         # generic HF-like micro
make mixtral-micro    # Mixtral names + convert + probe
make stream-mixtral   # streaming path on Mixtral micro
make stream-qwen      # streaming path on Qwen micro
make m2               # convert + C↔NumPy parity
```

## Real checkpoint

```bash
pip install safetensors numpy
python tools/convert_hf_stream.py \
  --model-dir /path/to/hf \
  --out models/real-v2.rabbit \
  --quant 4 --name real-moe
./rabbit chat --model models/real-v2.rabbit
```

Large models still need aggressive expert limits / quant; v2 is about **shape fidelity**, not RAM magic.

See [PHASE_D.md](PHASE_D.md).
