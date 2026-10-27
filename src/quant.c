#include "rabbit.h"
#include "cuda_gemv.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef RABBIT_INT4_GROUP
#define RABBIT_INT4_GROUP 32
#endif

void rabbit_dequant_int8(const int8_t *src, float *dst, int n, float scale) {
    if (!src || !dst || n <= 0) return;
    for (int i = 0; i < n; i++)
        dst[i] = (float)src[i] * scale;
}

void rabbit_dequant_int4(const uint8_t *src, float *dst, int n, float scale) {
    if (!src || !dst || n <= 0) return;

    int pairs = n / 2;
    for (int i = 0; i < pairs; i++) {
        uint8_t b = src[i];
        int lo = (int)(b & 0x0F) - 8;
        int hi = (int)((b >> 4) & 0x0F) - 8;
        dst[i * 2]     = (float)lo * scale;
        dst[i * 2 + 1] = (float)hi * scale;
    }
    if (n & 1) {
        uint8_t b = src[pairs];
        int lo = (int)(b & 0x0F) - 8;
        dst[n - 1] = (float)lo * scale;
    }
}

static int clamp_i(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

int rabbit_pack_int4(const float *src, int n, uint8_t *dst, float *out_scale) {
    if (!src || !dst || n <= 0) return -1;

    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = src[i] < 0.0f ? -src[i] : src[i];
        if (a > max_abs) max_abs = a;
    }
    float scale = max_abs / 7.0f + 1e-8f;
    if (out_scale) *out_scale = scale;

    int pairs = n / 2;
    for (int i = 0; i < pairs; i++) {
        int lo = clamp_i((int)lrintf(src[i * 2] / scale), -8, 7);
        int hi = clamp_i((int)lrintf(src[i * 2 + 1] / scale), -8, 7);
        dst[i] = (uint8_t)((lo + 8) | ((hi + 8) << 4));
    }
    if (n & 1) {
        int lo = clamp_i((int)lrintf(src[n - 1] / scale), -8, 7);
        dst[pairs] = (uint8_t)(lo + 8);
    }
    return (n + 1) / 2;
}

int rabbit_pack_int4_grouped(const float *src, int n, int group_size,
                             uint8_t *dst, int dst_cap) {
    if (!src || !dst || n <= 0) return -1;
    if (group_size <= 0) group_size = RABBIT_INT4_GROUP;

    int n_groups = (n + group_size - 1) / group_size;
    int packed_bytes = (n + 1) / 2;
    int need = 4 + n_groups * 4 + packed_bytes;
    if (dst_cap < need) return -1;

    uint8_t *p = dst;
    uint32_t gs = (uint32_t)group_size;
    memcpy(p, &gs, 4);
    p += 4;

    float *scales = (float *)p;
    p += n_groups * 4;

    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int end = start + group_size;
        if (end > n) end = n;
        float max_abs = 0.0f;
        for (int i = start; i < end; i++) {
            float a = src[i] < 0.0f ? -src[i] : src[i];
            if (a > max_abs) max_abs = a;
        }
        scales[g] = max_abs / 7.0f + 1e-8f;
    }

    for (int i = 0; i < n; i += 2) {
        int g0 = i / group_size;
        float s0 = scales[g0];
        int lo = clamp_i((int)lrintf(src[i] / s0), -8, 7);
        int hi = 0;
        if (i + 1 < n) {
            int g1 = (i + 1) / group_size;
            float s1 = scales[g1];
            hi = clamp_i((int)lrintf(src[i + 1] / s1), -8, 7);
        }
        p[i / 2] = (uint8_t)((lo + 8) | ((hi + 8) << 4));
    }
    return need;
}

int rabbit_dequant_int4_grouped(const uint8_t *src, float *dst, int n) {
    if (!src || !dst || n <= 0) return -1;

    uint32_t group_size = 0;
    memcpy(&group_size, src, 4);
    if (group_size == 0 || group_size > 4096) return -1;

    int n_groups = (n + (int)group_size - 1) / (int)group_size;
    const float *scales = (const float *)(src + 4);
    const uint8_t *packed = src + 4 + n_groups * 4;

    for (int i = 0; i < n; i++) {
        int byte_i = i / 2;
        int nibble = (i & 1) ? ((packed[byte_i] >> 4) & 0x0F) : (packed[byte_i] & 0x0F);
        int q = nibble - 8;
        int g = i / (int)group_size;
        dst[i] = (float)q * scales[g];
    }
    return 0;
}

static void gemv_int4_grouped_cpu(const uint8_t *W, const float *x, float *y,
                                  int out_dim, int in_dim) {
    int n = out_dim * in_dim;
    float *full = (float *)malloc(sizeof(float) * (size_t)n);
    if (!full) {
        memset(y, 0, sizeof(float) * (size_t)out_dim);
        return;
    }
    if (rabbit_dequant_int4_grouped(W, full, n) != 0) {
        free(full);
        memset(y, 0, sizeof(float) * (size_t)out_dim);
        return;
    }
    for (int o = 0; o < out_dim; o++) {
        float sum = 0.0f;
        const float *row = full + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; i++)
            sum += row[i] * x[i];
        y[o] = sum;
    }
    free(full);
}

void rabbit_gemv_int4_grouped(const uint8_t *W, const float *x, float *y,
                              int out_dim, int in_dim) {
    if (!W || !x || !y || out_dim <= 0 || in_dim <= 0) return;

    /* Optional CUDA path (no-op stub when RABBIT_CUDA not enabled). */
    if (rabbit_cuda_available() &&
        rabbit_cuda_gemv_int4_grouped(W, x, y, out_dim, in_dim) == 0)
        return;

    gemv_int4_grouped_cpu(W, x, y, out_dim, in_dim);
}

int rabbit_int4_grouped_nbytes(int n, int group_size) {
    if (n <= 0) return 0;
    if (group_size <= 0) group_size = RABBIT_INT4_GROUP;
    int n_groups = (n + group_size - 1) / group_size;
    return 4 + n_groups * 4 + (n + 1) / 2;
}
