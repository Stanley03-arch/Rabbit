#include "rabbit.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

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
