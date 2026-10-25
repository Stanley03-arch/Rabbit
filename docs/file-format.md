# Rabbit file format (v1)

Little-endian. Designed for a single file and simple `mmap` of the expert region.

## Layout

```
[ Header ]
[ Dense blob ]
[ Expert blob ]   # n_expert_records contiguous expert records
```

## Header

Fixed struct (`RabbitHeader` in `src/rabbit.h`):

| Field | Type | Description |
|-------|------|-------------|
| magic | `char[8]` | `"RABBIT01"` |
| version | `u32` | `1` |
| header_size | `u32` | Size of this header |
| n_layers | `u32` | Transformer layers |
| hidden_dim | `u32` | Model width |
| vocab_size | `u32` | Vocabulary size (metadata) |
| n_experts | `u32` | Number of routed experts |
| experts_per_token | `u32` | Top-k |
| expert_dim | `u32` | Intermediate size (metadata) |
| dense_size | `u64` | Bytes in dense blob |
| expert_size | `u64` | Bytes **per** expert record |
| total_experts_size | `u64` | `n_expert_records * expert_size` |
| dense_quant | `u32` | 0/16 ≈ fp16-ish metadata; 8 = int8 |
| expert_quant | `u32` | **0** = f32, **4** = group-scaled int4, **8** = int8 |
| name | `char[64]` | Human-readable id |
| reserved | `u8[128]` | Zero for now |

## Dense blob

Opaque to the generic loader: entire region is read into RAM. Internal layout is model-family specific (v2 packs embed, lm_head, per-layer norms/QKV/O/router).

## Expert record (v2 / SwiGLU)

Three matrices in order: **gate [I,H]**, **up [I,H]**, **down [H,I]**.

### expert_quant = 0 (float32)

Raw little-endian `float32` payloads concatenated.

### expert_quant = 8 (int8)

Per matrix:

```
float32 scale
int8 weights[rows * cols]
```

### expert_quant = 4 (group-scaled int4)

Per matrix (`n = rows * cols` elements):

```
u32  group_size          # default 32
f32  scales[n_groups]    # n_groups = ceil(n / group_size)
u8   packed[(n+1)/2]     # two signed 4-bit values per byte, bias +8
```

Matches `rabbit_pack_int4_grouped` / `rabbit_dequant_int4_grouped` in `src/quant.c`.

Records are padded to a uniform `expert_size` so indexing is `base + id * expert_size`.

## Compatibility

Older synthetic test models may omit the scale header and store a raw int8 matrix. The engine still accepts that path for development.
