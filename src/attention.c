#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Phase B attention:
 *   - multi-head over a sliding KV window
 *   - RoPE on Q and K (per head)
 *   - residual mix + lightweight dense residual from dense blob
 */

#define RABBIT_MAX_CTX 256
#define RABBIT_HEAD_DIM 32

typedef struct {
    float *k_cache; /* [max_len, dim] */
    float *v_cache;
    int    len;
    int    dim;
    int    max_len;
    int    pos;     /* absolute position for RoPE */
} TinyKVCache;

static TinyKVCache g_kv = {0};

void rabbit_attn_reset(void) {
    free(g_kv.k_cache);
    free(g_kv.v_cache);
    memset(&g_kv, 0, sizeof(g_kv));
}

static int ensure_kv(int dim) {
    if (g_kv.k_cache && g_kv.dim == dim) return 0;

    rabbit_attn_reset();
    g_kv.dim = dim;
    g_kv.max_len = RABBIT_MAX_CTX;
    g_kv.k_cache = calloc((size_t)g_kv.max_len * dim, sizeof(float));
    g_kv.v_cache = calloc((size_t)g_kv.max_len * dim, sizeof(float));
    if (!g_kv.k_cache || !g_kv.v_cache) {
        rabbit_attn_reset();
        return -1;
    }
    return 0;
}

/* Apply RoPE in-place to a single head vector of length head_dim. */
static void rope_apply(float *q, float *k, int head_dim, int pos) {
    int half = head_dim / 2;
    if (half < 1) return;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(10000.0f, (2.0f * (float)i) / (float)head_dim);
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float q0 = q[i], q1 = q[i + half];
        float k0 = k[i], k1 = k[i + half];
        q[i] = q0 * c - q1 * s;
        q[i + half] = q0 * s + q1 * c;
        k[i] = k0 * c - k1 * s;
        k[i + half] = k0 * s + k1 * c;
    }
}

static void multihead_attention(float *x, int dim) {
    if (ensure_kv(dim) != 0) return;
    if (dim > 2048) return;

    int head_dim = RABBIT_HEAD_DIM;
    if (head_dim > dim) head_dim = dim;
    int n_heads = dim / head_dim;
    if (n_heads < 1) {
        n_heads = 1;
        head_dim = dim;
    }
    int used = n_heads * head_dim;

    /* Build Q from x; K/V = RoPE(x) then store */
    float *q = (float *)malloc((size_t)dim * sizeof(float));
    float *k_rot = (float *)malloc((size_t)dim * sizeof(float));
    float *out = (float *)calloc((size_t)dim, sizeof(float));
    if (!q || !k_rot || !out) {
        free(q); free(k_rot); free(out);
        return;
    }
    memcpy(q, x, (size_t)dim * sizeof(float));
    memcpy(k_rot, x, (size_t)dim * sizeof(float));

    int pos = g_kv.pos;
    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
        float *kh = k_rot + h * head_dim;
        rope_apply(qh, kh, head_dim, pos);
    }

    /* Append to KV cache (store rotated K, raw V = x) */
    if (g_kv.len < g_kv.max_len) {
        memcpy(g_kv.k_cache + g_kv.len * dim, k_rot, (size_t)dim * sizeof(float));
        memcpy(g_kv.v_cache + g_kv.len * dim, x, (size_t)dim * sizeof(float));
        g_kv.len++;
    } else {
        memmove(g_kv.k_cache, g_kv.k_cache + dim,
                (size_t)(g_kv.max_len - 1) * dim * sizeof(float));
        memmove(g_kv.v_cache, g_kv.v_cache + dim,
                (size_t)(g_kv.max_len - 1) * dim * sizeof(float));
        memcpy(g_kv.k_cache + (g_kv.max_len - 1) * dim, k_rot, (size_t)dim * sizeof(float));
        memcpy(g_kv.v_cache + (g_kv.max_len - 1) * dim, x, (size_t)dim * sizeof(float));
    }
    g_kv.pos++;

    float scale = 1.0f / sqrtf((float)head_dim);
    float scores[RABBIT_MAX_CTX];

    for (int h = 0; h < n_heads; h++) {
        int off = h * head_dim;
        for (int t = 0; t < g_kv.len; t++) {
            float dot = 0.0f;
            const float *krow = g_kv.k_cache + t * dim + off;
            for (int i = 0; i < head_dim; i++)
                dot += q[off + i] * krow[i];
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
            for (int i = 0; i < head_dim; i++)
                out[off + i] += scores[t] * v[i];
        }
    }

    /* Residual for used dims; pass-through unused tail */
    for (int i = 0; i < used; i++)
        x[i] = 0.5f * x[i] + 0.5f * out[i];

    free(q); free(k_rot); free(out);
}

static void tiny_dense_ffn(RabbitModel *m, float *x, int dim) {
    if (!m || !m->dense || m->dense_size < (size_t)dim) return;

    const uint8_t *bytes = (const uint8_t *)m->dense;
    float scale = 1.0f / 128.0f;

    static int offset = 0;
    offset = (offset + dim * 3) % (int)(m->dense_size > 0 ? m->dense_size : 1);

    float *tmp = (float *)malloc((size_t)dim * sizeof(float));
    if (!tmp) return;

    for (int i = 0; i < dim; i++) {
        int idx = (offset + i) % (int)m->dense_size;
        float w = ((float)bytes[idx] - 128.0f) * scale;
        float gate = 1.0f / (1.0f + expf(-x[i]));
        tmp[i] = x[i] + 0.15f * w * gate * x[(i * 7) % dim];
    }
    memcpy(x, tmp, (size_t)dim * sizeof(float));
    free(tmp);
}

int rabbit_layer_step(RabbitModel *m, float *hidden, int hidden_dim) {
    if (!m || !hidden || hidden_dim <= 0) return -1;
    multihead_attention(hidden, hidden_dim);
    tiny_dense_ffn(m, hidden, hidden_dim);
    return rabbit_moe_step(m, hidden, hidden_dim);
}

size_t rabbit_kv_bytes_estimate(int hidden_dim, int max_ctx) {
    if (hidden_dim <= 0) hidden_dim = 1;
    if (max_ctx <= 0) max_ctx = RABBIT_MAX_CTX;
    /* K + V float32 */
    return (size_t)max_ctx * (size_t)hidden_dim * sizeof(float) * 2;
}
