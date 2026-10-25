/* Unit tests for quant pack/dequant round-trip */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "rabbit.h"

static int g_fail = 0;

static void expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_fail++;
    }
}

static void test_int4_roundtrip(void) {
    const int n = 17;
    float src[17];
    float dst[17];
    uint8_t packed[16];

    for (int i = 0; i < n; i++)
        src[i] = (float)(i - 8) * 0.3f;

    float scale = 0.0f;
    int nbytes = rabbit_pack_int4(src, n, packed, &scale);
    expect(nbytes > 0, "pack_int4 returns bytes");
    expect(scale > 0.0f, "scale positive");

    rabbit_dequant_int4(packed, dst, n, scale);

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(src[i] - dst[i]);
        if (e > max_err) max_err = e;
    }
    expect(max_err < 1.0f, "int4 round-trip error bound");
}

static void test_int4_grouped_roundtrip(void) {
    const int n = 100;
    float src[100];
    float dst[100];
    uint8_t buf[512];

    for (int i = 0; i < n; i++)
        src[i] = sinf((float)i * 0.17f) * 3.0f;

    int written = rabbit_pack_int4_grouped(src, n, 32, buf, (int)sizeof(buf));
    expect(written > 0, "pack_int4_grouped writes");
    expect(written == rabbit_int4_grouped_nbytes(n, 32), "nbytes matches");

    expect(rabbit_dequant_int4_grouped(buf, dst, n) == 0, "dequant ok");

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(src[i] - dst[i]);
        if (e > max_err) max_err = e;
    }
    /* group scale is tighter than global; still coarse */
    expect(max_err < 0.6f, "group int4 error bound");
}

static void test_int8_dequant(void) {
    int8_t src[4] = {-128, -1, 0, 127};
    float dst[4];
    rabbit_dequant_int8(src, dst, 4, 0.5f);
    expect(fabsf(dst[0] - (-64.0f)) < 1e-5f, "int8 dequant -128");
    expect(fabsf(dst[3] - 63.5f) < 1e-5f, "int8 dequant 127");
}

static void test_tokenize_byte(void) {
    rabbit_tokenizer_set_byte_mode();
    int ids[8];
    int n = rabbit_tokenize("AB", ids, 8);
    expect(n == 2, "tokenize length");
    expect(ids[0] == (int)'A' && ids[1] == (int)'B', "tokenize values");
}

int main(void) {
    test_int4_roundtrip();
    test_int4_grouped_roundtrip();
    test_int8_dequant();
    test_tokenize_byte();

    if (g_fail) {
        fprintf(stderr, "%d test(s) failed\n", g_fail);
        return 1;
    }
    printf("ok: quant + tokenizer unit tests passed\n");
    return 0;
}
