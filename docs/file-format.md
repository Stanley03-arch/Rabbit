# Rabbit Model File Format (v0)

Simple, mmap-friendly format designed for constrained hardware.

## Overall Layout

```
+---------------------------+
| Header (fixed size)       |
+---------------------------+
| Dense blob                |  <-- always loaded into RAM
+---------------------------+
| Expert blob               |  <-- memory-mapped, streamed
+---------------------------+
```

## Header (little-endian)

```c
struct RabbitHeader {
    char     magic[8];          // "RABBIT01"
    uint32_t version;           // 1
    uint32_t header_size;       // sizeof(RabbitHeader)

    // Architecture
    uint32_t n_layers;
    uint32_t hidden_dim;
    uint32_t vocab_size;
    uint32_t n_experts;
    uint32_t experts_per_token;
    uint32_t expert_dim;        // intermediate size of one expert

    // Sizes in bytes
    uint64_t dense_size;
    uint64_t expert_size;       // size of ONE expert
    uint64_t total_experts_size;// n_experts * expert_size

    // Quantization
    uint32_t dense_quant;       // 0 = f16, 4 = int4, etc.
    uint32_t expert_quant;

    char     name[64];          // human name
    uint8_t  reserved[128];     // future use
};
```

Magic must be exactly `RABBIT01`.

## Dense Blob

Contains everything that must stay in RAM:
- token embeddings
- attention weights (all layers)
- shared experts (if any)
- layer norms, output head, etc.

Layout inside the dense blob is model-family specific and documented later.

## Expert Blob

Contiguous array of experts:

```
expert[0]
expert[1]
...
expert[n_experts-1]
```

Each expert is exactly `expert_size` bytes and is aligned for fast `pread` / mmap access.

## Design Goals

- Single file (easy distribution)
- Dense part small enough for 4 GB machines
- Experts can be streamed with zero-copy mmap
- Simple enough to implement in pure C with no external libraries
