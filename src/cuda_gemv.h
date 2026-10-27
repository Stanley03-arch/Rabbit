#ifndef RABBIT_CUDA_GEMV_H
#define RABBIT_CUDA_GEMV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 if a CUDA device is present and the real backend was linked. */
int rabbit_cuda_available(void);

/*
 * Device-accelerated y = W @ x for one group-scaled int4 matrix.
 * W layout matches CPU rabbit_gemv_int4_grouped / pack_int4_grouped:
 *   u32 group_size | f32 scales[n_groups] | u8 packed[(n+1)/2]
 * with n = out_dim * in_dim, row-major.
 *
 * Host pointers in/out. Returns 0 on success, -1 to fall back to CPU.
 */
int rabbit_cuda_gemv_int4_grouped(const uint8_t *W, const float *x, float *y,
                                  int out_dim, int in_dim);

#ifdef __cplusplus
}
#endif

#endif /* RABBIT_CUDA_GEMV_H */
