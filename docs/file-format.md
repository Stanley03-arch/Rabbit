# Rabbit file format (v1)

Little-endian. Designed for a single file and simple `mmap` of the expert region.

## Layout

```
[ Header ]
[ Dense blob ]
[ Expert blob ]   # n_experts contiguous expert records
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
| total_experts_size | `u64` | `n_experts * expert_size` |
| dense_quant | `u32` | 0/16 ≈ fp16-ish metadata; 8 = int8 |
| expert_quant | `u32` | 4 or 8 |
| name | `char[64]` | Human-readable id |
| reserved | `u8[128]` | Zero for now |

## Dense blob

Opaque to the generic loader: entire region is read into RAM. Internal layout is model-family specific.

## Expert record

Preferred layout written by the converter:

```
float32 scale
packed weights   # int8 or int4 according to expert_quant
```

Records are padded to a uniform `expert_size` so indexing is `base + id * expert_size`.

## Compatibility

Older synthetic test models may omit the scale header and store a raw int8 matrix. The engine still accepts that path for development.
