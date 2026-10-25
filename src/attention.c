#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Minimal attention + residual dense path for the synthetic test model.
 *
 * This is intentionally simple so it fits the 4 GB design goals and
 * can later be replaced by real multi-head / MLA / KDA implementations.
 *
 * For the tiny model we treat the dense blob as a pool of weights and
 * implement:
 *   - a single-head self-attention over a short context window
 *   - a residual feed-forward style update using dense weights
 */

#define RABBIT_MAX_CTX 64

typedef struct {
    float *k_cache;   /* [ctx, dim] */
    float *v_cache;
    int    len;
    int    dim;
    int    max_len;
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
    g_kv.k_cache = calloc(g_kv.max_len * dim, sizeof(float));
    g_kv.v_cache = calloc(g_kv.max_len * dim, sizeof(float));
    if (!g_kv.k_cache || !g_kv.v_cache) {
        rabbit_attn_reset();
        return -1;
    }
    return 0;
}

/* Simple scaled dot-product attention over the growing KV cache */
static void tiny_attention(float *x, int dim) {
    if (ensure_kv(dim) != 0) return;

    /* Append current x as new key/value */
    if (g_kv.len < g_kv.max_len) {
        memcpy(g_kv.k_cache + g_kv.len * dim, x, dim * sizeof(float));
        memcpy(g_kv.v_cache + g_kv.len * dim, x, dim * sizeof(float));
        g_kv.len++;
    } else {
        /* Shift left (drop oldest) */
        memmove(g_kv.k_cache, g_kv.k_cache + dim, (g_kv.max_len - 1) * dim * sizeof(float));
        memmove(g_kv.v_cache, g_kv.v_cache + dim, (g_kv.max_len - 1) * dim * sizeof(float));
        memcpy(g_kv.k_cache + (g_kv.max_len - 1) * dim, x, dim * sizeof(float));
        memcpy(g_kv.v_cache + (g_kv.max_len - 1) * dim, x, dim * sizeof(float));
    }

    /* Compute attention scores against all keys */
    float scores[RABBIT_MAX_CTX];
    float scale = 1.0f / sqrtf((float)dim);

    for (int t = 0; t < g_kv.len; t++) {
        float dot = 0.0f;
        const float *k = g_kv.k_cache + t * dim;
        for (int i = 0; i < dim; i++)
            dot += x[i] * k[i];
        scores[t] = dot * scale;
    }

    /* Softmax */
    float max_s = scores[0];
    for (int t = 1; t < g_kv.len; t++)
        if (scores[t] > max_s) max_s = scores[t];

    float sum = 0.0f;
    for (int t = 0; t < g_kv.len; t++) {
        scores[t] = expf(scores[t] - max_s);
        sum += scores[t];
    }
    if (sum > 0.0f) {
        for (int t = 0; t < g_kv.len; t++)
            scores[t] /= sum;
    }

    /* Weighted sum of values → write back into x (residual style) */
    float out[512];
    if (dim > 512) return;
    memset(out, 0, dim * sizeof(float));

    for (int t = 0; t < g_kv.len; t++) {
        const float *v = g_kv.v_cache + t * dim;
        for (int i = 0; i < dim; i++)
            out[i] += scores[t] * v[i];
    }

    for (int i = 0; i < dim; i++)
        x[i] = x[i] * 0.5f + out[i] * 0.5f;   /* residual blend */
}

/*
 * Dense residual update using a slice of the dense blob.
 * Treats dense weights as a simple linear transform.
 */
static void tiny_dense_ffn(RabbitModel *m, float *x, int dim) {
    if (!m || !m->dense || m->dense_size < (size_t)dim) return;

    const uint8_t *bytes = (const uint8_t *)m->dense;
    float scale = 1.0f / 128.0f;

    /* Use a rotating window into the dense blob so different layers
       see slightly different weights. */
    static int offset = 0;
    offset = (offset + dim) % (int)(m->dense_size > 0 ? m->dense_size : 1);

    float tmp[512];
    if (dim > 512) return;

    for (int i = 0; i < dim; i++) {
        int idx = (offset + i) % (int)m->dense_size;
        float w = ((float)bytes[idx] - 128.0f) * scale;
        tmp[i] = x[i] + 0.1f * w * x[i % (dim > 8 ? 8 : dim)];
    }

    memcpy(x, tmp, dim * sizeof(float));
}

/*
 * One full layer: attention → residual FFN → MoE
 */
int rabbit_layer_step(RabbitModel *m, float *hidden, int hidden_dim) {
    if (!m || !hidden || hidden_dim <= 0) return -1;

    tiny_attention(hidden, hidden_dim);
    tiny_dense_ffn(m, hidden, hidden_dim);
    return rabbit_moe_step(m, hidden, hidden_dim);
}
