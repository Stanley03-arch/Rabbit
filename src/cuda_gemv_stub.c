/* CPU-only stub — linked when RABBIT_CUDA is not enabled. */
#include "cuda_gemv.h"

int rabbit_cuda_available(void) {
    return 0;
}

int rabbit_cuda_gemv_int4_grouped(const uint8_t *W, const float *x, float *y,
                                  int out_dim, int in_dim) {
    (void)W;
    (void)x;
    (void)y;
    (void)out_dim;
    (void)in_dim;
    return -1;
}
