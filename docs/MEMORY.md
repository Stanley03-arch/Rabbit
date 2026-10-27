# Multitier memory model

Rabbit keeps **dense** weights in RAM and **routed experts** on disk via `mmap`, with a small **RAM LRU** for hot experts. This is intentional for ~4 GB hosts — not a full VRAM/NVMe “learned placement” system.

## Tiers

| Tier | What | Mechanism |
|------|------|-----------|
| **RAM dense** | embed, norms, QKV, routers, … | `malloc` at load |
| **RAM expert LRU** | last-used expert blobs | `rabbit_cache_*`, slots from budget |
| **Disk / page cache** | all expert records | `mmap` + `MADV_RANDOM` |
| **OS hints** | optional warm/cool | `madvise(WILLNEED)` / `DONTNEED` |

Optional **CUDA** (`make RABBIT_CUDA=1`) only accelerates group-int4 GEMV; it does not stage the full expert table in VRAM.

## Knobs

```bash
export RABBIT_MEM_BUDGET_MB=1024    # default 1536
export RABBIT_CACHE_SLOTS=8         # override auto slot count (1..16)
```

Slot count is chosen from remaining budget after dense weights unless `RABBIT_CACHE_SLOTS` is set.

## APIs

```c
rabbit_fetch_expert(m, id, promote);   /* LRU then mmap */
rabbit_expert_prefetch(m, id);         /* MADV_WILLNEED on mmap page */
rabbit_expert_release_hint(m, id);     /* MADV_DONTNEED */
rabbit_cache_pin(cache, id, 1);        /* protect slot from eviction */
```

`./rabbit stats --model …` reports dense/cache/KV estimates and cache hit counters.

## Non-goals (this release)

- Learned expert placement across VRAM/RAM/NVMe
- `O_DIRECT` I/O path (page cache is the disk tier for now)
- Uploading every expert to GPU at init
