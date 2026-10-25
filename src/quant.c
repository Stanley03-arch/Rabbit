#include "rabbit.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/*
 * Quantization / dequantization helpers.
 * First version focuses on simple paths that the MoE kernels can call.
 */

/* Treat a block of bytes as int8 and dequantize to float */
void rabbit_dequant_int8(const int8_t *src, float *dst, int n, float scale) {
    for (int i = 0; i < n; i++)
        dst[i] = (float)src[i] * scale;
}

/* Very simple int4 unpack (two values per byte, low nibble first) */
void rabbit_dequant_int4(const uint8_t *src, float *dst, int n, float scale) {
    int pairs = n / 2;
    for (int i = 0; i < pairs; i++) {
        uint8_t b = src[i];
        int lo = (b & 0x0F) - 8;          /* -8 .. 7 */
        int hi = ((b >> 4) & 0x0F) - 8;
        dst[i*2]     = (float)lo * scale;
        dst[i*2 + 1] = (float)hi * scale;
    }
    if (n & 1) {
        uint8_t b = src[pairs];
        int lo = (b & 0x0F) - 8;
        dst[n-1] = (float)lo * scale;
    }
}

void rabbit_quant_stub(void) {
    printf("[quant] int8 and int4 dequant helpers are available.\n");
}
