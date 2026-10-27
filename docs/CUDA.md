# Optional CUDA GEMV

Rabbit stays **CPU/`mmap` first**. CUDA is an optional accelerator for **group-scaled int4 GEMV** only.

## Build

```bash
# Default — no CUDA toolkit required
make clean && make

# With CUDA (nvcc + libcudart)
make clean && make RABBIT_CUDA=1
```

## Behavior

- `rabbit_gemv_int4_grouped` calls `rabbit_cuda_gemv_int4_grouped` when a device is available.
- On failure or stub builds, the existing CPU path runs unchanged.
- Experts are **not** bulk-uploaded to VRAM at init (that would fight the disk-backed design). Each GEMV may copy one matrix for the kernel; further caching can be added later.

## Files

| File | Role |
|------|------|
| `src/cuda_gemv.h` | C API |
| `src/cuda_gemv_stub.c` | Default link |
| `src/cuda_gemv.cu` | Real kernel (`RABBIT_CUDA=1`) |

## Not goals (yet)

- Full MoE / attention on GPU
- Multitier VRAM/RAM/NVMe planner
