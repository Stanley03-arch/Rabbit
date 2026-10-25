#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Multi-head attention + residual dense path (Phase B).
 *
 * - Configurable number of heads derived from hidden_dim
 * - Per-head RoPE
 * - Short KV window (max 128)
 * - Residual dense FFN using the dense blob
 */

#define RABBIT_MAX_CTX 128
#define RABBIT_MAX_HEADS 16
#define RABBIT_MIN_HEAD_DIM 8

typedef struct {
    float *k_cache;   /* [max_len * dim] */
    float *v_cache;
    int    len;
    int    dim;
    int    max_len;
    int    pos;
    int    n_heads;
    int    head_dim;
} TinyKVCache;

static TinyKVCache g_kv = {0};

void rabbit_attn_reset(void) {
    free(g_kv.k_cache);
    free(g_kv.v_cache);
    memset(&g_kv, 0, sizeof(g_kv));
}

static void derive_heads(int dim, int *n_heads, int *head_dim) {
    int h = 1;
    if (dim >= 64) h = 4;
    if (dim >= 128) h = 8;
    if (dim >= 256) h = 16;
    if (h > RABBIT_MAX_HEADS) h = RABBIT_MAX_HEADS;
    while (h > 1 && (dim / h) < RABBIT_MIN_HEAD_DIM)
        h /= 2;
    if (h < 1) h = 1;
    *n_heads = h;
    *head_dim = dim / h;
}

static int ensure_kv(int dim) {
    if (g_kv.k_cache && g_kv.dim == dim) return 0;

    rabbit_attn_reset();
    g_kv.dim = dim;
    g_kv.max_len = RABBIT_MAX_CTX;
    derive_heads(dim, &g_kv.n_heads, &g_kv.head_dim);

    g_kv.k_cache = calloc((size_t)g_kv.max_len * dim, sizeof(float));
    g_kv.v_cache = calloc((size_t)g_kv.max_len * dim, sizeof(float));
    if (!g_kv.k_cache || !g_kv.v_cache) {
        rabbit_attn_reset();
        return -1;
    }
    return 0;
}

/* RoPE on a single head (head_dim elements). */
static void apply_rope_head(float *x, int head_dim, int pos) {
    if (head_dim < 2) return;
    const float theta = 10000.0f;
    int pairs = head_dim / 2;
    for (int i = 0; i < pairs; i++) {
        float freq = 1.0f / powf(theta, (2.0f * (float)i) / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float x0 = x[2 * i];
        float x1 = x[2 * i + 1];
        x[2 * i]     = x0 * cos_a - x1 * sin_a;
        x[2 * i + 1] = x0 * sin_a + x1 * cos_a;
    }
}

static void tiny_attention(float *x, int dim) {
    if (ensure_kv(dim) != 0) return;

    float *q = (float *)malloc((size_t)dim * sizeof(float));
    if (!q) return;
    memcpy(q, x, (size_t)dim * sizeof(float));

    /* RoPE per head on query */
    for (int h = 0; h < g_kv.n_heads; h++)
        apply_rope_head(q + h * g_kv.head_dim, g_kv.head_dim, g_kv.pos);

    if (g_kv.len < g_kv.max_len) {
        /* Store rotated K and original V */
        memcpy(g_kv.k_cache + g_kv.len * dim, q, (size_t)dim * sizeof(float));
        memcpy(g_kv.v_cache + g_kv.len * dim, x, (size_t)dim * sizeof(float));
        g_kv.len++;
    } else {
        memmove(g_kv.k_cache, g_kv.k_cache + dim,
                (size_t)(g_kv.max_len - 1) * dim * sizeof(float));
        memmove(g_kv.v_cache, g_kv.v_cache + dim,
                (size_t)(g_kv.max_len - 1) * dim * sizeof(float));
        memcpy(g_kv.k_cache + (g_kv.max_len - 1) * dim, q, (size_t)dim * sizeof(float));
        memcpy(g_kv.v_cache + (g_kv.max_len - 1) * dim, x, (size_t)dim * sizeof(float));
    }
    g_kv.pos++;

    float scale = 1.0f / sqrtf((float)g_kv.head_dim);
    float scores[RABBIT_MAX_CTX];
    float out[512];
    if (dim > 512) {
        free(q);
        return;
    }
    memset(out, 0, (size_t)dim * sizeof(float));

    for (int h = 0; h < g_kv.n_heads; h++) {
        int off = h * g_kv.head_dim;
        int hd = g_kv.head_dim;

        for (int t = 0; t < g_kv.len; t++) {
            float dot = 0.0f;
            const float *k = g_kv.k_cache + t * dim + off;
            for (int i = 0; i < hd; i++)
                dot += q[off + i] * k[i];
            scores[t] = dot * scale;
        }

        float max_s = scores[0];
        for (int t = 1; t < g_kv.len; t++)
            if (scores[t] > max_s) max_s = scores[t];

        float sum = 0.0f;
        for (int t = 0; t < g_kv.len; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum += scores[t];
        }
        if (sum > 0.0f)
            for (int t = 0; t < g_kv.len; t++)
                scores[t] /= sum;

        for (int t = 0; t < g_kv.len; t++) {
            const float *v = g_kv.v_cache + t * dim + off;
            for (int i = 0; i < hd; i++)
                out[off + i] += scores[t] * v[i];
        }
    }

    for (int i = 0; i < dim; i++)
        x[i] = x[i] * 0.5f + out[i] * 0.5f;

    free(q);
}

static void tiny_dense_ffn(RabbitModel *m, float *x, int dim) {
    if (!m || !m->dense || m->dense_size < (size_t)dim) return;

    const uint8_t *bytes = (const uint8_t *)m->dense;
    float scale = 1.0f / 128.0f;

    static int offset = 0;
    offset = (offset + dim * 3) % (int)(m->dense_size > 0 ? m->dense_size : 1);

    float tmp[512];
    if (dim > 512) return;

    for (int i = 0; i < dim; i++) {
        int idx = (offset + i) % (int)m->dense_size;
        float w = ((float)bytes[idx] - 128.0f) * scale;
        float gate = 1.0f / (1.0f + expf(-x[i]));
        tmp[i] = x[i] + 0.15f * w * gate * x[(i * 7) % dim];
    }
    memcpy(x, tmp, (size_t)dim * sizeof(float));
}

int rabbit_layer_step(RabbitModel *m, float *hidden, int hidden_dim) {
    if (!m || !hidden || hidden_dim <= 0) return -1;
    tiny_attention(hidden, hidden_dim);
    tiny_dense_ffn(m, hidden, hidden_dim);
    return rabbit_moe_step(m, hidden, hidden_dim);
}
