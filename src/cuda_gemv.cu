/*
 * Optional CUDA backend for group-scaled int4 GEMV.
 * Build: make RABBIT_CUDA=1
 */
#include "cuda_gemv.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

#ifndef RABBIT_INT4_GROUP
#define RABBIT_INT4_GROUP 32
#endif

__global__ void gemv_int4_grouped_kernel(const uint8_t *__restrict__ W,
                                         const float *__restrict__ x,
                                         float *__restrict__ y,
                                         int out_dim, int in_dim) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim)
        return;

    uint32_t group_size;
    memcpy(&group_size, W, 4);
    if (group_size == 0 || group_size > 4096u) {
        y[o] = 0.0f;
        return;
    }

    int n = out_dim * in_dim;
    int n_groups = (n + (int)group_size - 1) / (int)group_size;
    const float *scales = (const float *)(W + 4);
    const uint8_t *packed = W + 4 + n_groups * 4;

    float sum = 0.0f;
    int row_start = o * in_dim;
    for (int i = 0; i < in_dim; i++) {
        int idx = row_start + i;
        int byte_i = idx >> 1;
        int nibble = (idx & 1) ? ((packed[byte_i] >> 4) & 0x0F) : (packed[byte_i] & 0x0F);
        int q = nibble - 8;
        int g = idx / (int)group_size;
        sum += (float)q * scales[g] * x[i];
    }
    y[o] = sum;
}

extern "C" int rabbit_cuda_available(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess)
        return 0;
    return n > 0 ? 1 : 0;
}

extern "C" int rabbit_cuda_gemv_int4_grouped(const uint8_t *W, const float *x, float *y,
                                             int out_dim, int in_dim) {
    if (!W || !x || !y || out_dim <= 0 || in_dim <= 0)
        return -1;
    if (!rabbit_cuda_available())
        return -1;

    uint32_t group_size = 0;
    memcpy(&group_size, W, 4);
    if (group_size == 0 || group_size > 4096u)
        return -1;

    int n = out_dim * in_dim;
    int n_groups = (n + (int)group_size - 1) / (int)group_size;
    int packed_bytes = (n + 1) / 2;
    size_t W_bytes = (size_t)(4 + n_groups * 4 + packed_bytes);

    uint8_t *d_W = nullptr;
    float *d_x = nullptr;
    float *d_y = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_W, W_bytes);
    if (err != cudaSuccess)
        goto fail;
    err = cudaMalloc(&d_x, (size_t)in_dim * sizeof(float));
    if (err != cudaSuccess)
        goto fail;
    err = cudaMalloc(&d_y, (size_t)out_dim * sizeof(float));
    if (err != cudaSuccess)
        goto fail;

    err = cudaMemcpy(d_W, W, W_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        goto fail;
    err = cudaMemcpy(d_x, x, (size_t)in_dim * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        goto fail;

    {
        int threads = 256;
        int blocks = (out_dim + threads - 1) / threads;
        gemv_int4_grouped_kernel<<<blocks, threads>>>(d_W, d_x, d_y, out_dim, in_dim);
        err = cudaGetLastError();
        if (err != cudaSuccess)
            goto fail;
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
            goto fail;
    }

    err = cudaMemcpy(y, d_y, (size_t)out_dim * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        goto fail;

    cudaFree(d_W);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;

fail:
    if (d_W)
        cudaFree(d_W);
    if (d_x)
        cudaFree(d_x);
    if (d_y)
        cudaFree(d_y);
    return -1;
}
