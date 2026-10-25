#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Attention + residual dense path (Phase B improvements).
 *
 * - Short KV window with multi-group attention
 * - Basic RoPE on the query/key vectors
 * - Residual dense FFN using the dense blob
 */

#define RABBIT_MAX_CTX 128
#define RABBIT_N_GROUPS 4

typedef struct {
    float *k_cache;
    float *v_cache;
    int    len;
    int    dim;
    int    max_len;
    int    pos;          /* global position for RoPE */
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

/* Simple rotary embedding applied in-place to pairs of dimensions. */
static void apply_rope(float *x, int dim, int pos) {
    if (dim < 2) return;
    const float theta = 10000.0f;
    int pairs = dim / 2;
    for (int i = 0; i < pairs; i++) {
        float freq = 1.0f / powf(theta, (2.0f * i) / (float)dim);
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

    /* Work on a temporary so we can rotate without destroying residual */
    float *q = (float *)malloc((size_t)dim * sizeof(float));
    if (!q) return;
    memcpy(q, x, (size_t)dim * sizeof(float));
    apply_rope(q, dim, g_kv.pos);

    if (g_kv.len < g_kv.max_len) {
        memcpy(g_kv.k_cache + g_kv.len * dim, q, (size_t)dim * sizeof(float));
        memcpy(g_kv.v_cache + g_kv.len * dim, x, (size_t)dim * sizeof(float));
        g_kv.len++;
    } else {
        memmove(g_kv.k_cache, g_kv.k_cache + dim, (size_t)(g_kv.max_len - 1) * dim * sizeof(float));
        memmove(g_kv.v_cache, g_kv.v_cache + dim, (size_t)(g_kv.max_len - 1) * dim * sizeof(float));
        memcpy(g_kv.k_cache + (g_kv.max_len - 1) * dim, q, (size_t)dim * sizeof(float));
        memcpy(g_kv.v_cache + (g_kv.max_len - 1) * dim, x, (size_t)dim * sizeof(float));
    }
    g_kv.pos++;

    int groups = RABBIT_N_GROUPS;
    if (groups > dim) groups = 1;
    int group_dim = dim / groups;
    if (group_dim < 1) group_dim = dim;

    float scores[RABBIT_MAX_CTX];
    float scale = 1.0f / sqrtf((float)group_dim);

    float out[512];
    if (dim > 512) {
        free(q);
        return;
    }
    memset(out, 0, (size_t)dim * sizeof(float));

    for (int g = 0; g < groups; g++) {
        int off = g * group_dim;
        int gd = (g == groups - 1) ? (dim - off) : group_dim;

        for (int t = 0; t < g_kv.len; t++) {
            float dot = 0.0f;
            const float *k = g_kv.k_cache + t * dim + off;
            for (int i = 0; i < gd; i++)
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
            for (int i = 0; i < gd; i++)
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
