#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void gemv_int8(const int8_t *W, float scale, const float *x, float *y,
                      int out_dim, int in_dim, float weight) {
    for (int o = 0; o < out_dim; o++) {
        float sum = 0.0f;
        const int8_t *row = W + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; i++)
            sum += (float)row[i] * x[i];
        y[o] += weight * sum * scale;
    }
}

static void gemv_int4(const uint8_t *W, float scale, const float *x, float *y,
                      int out_dim, int in_dim, float weight) {
    int in_bytes = (in_dim + 1) / 2;
    for (int o = 0; o < out_dim; o++) {
        const uint8_t *row = W + (size_t)o * in_bytes;
        float sum = 0.0f;
        for (int i = 0; i < in_dim; i += 2) {
            uint8_t b = row[i / 2];
            int lo = (int)(b & 0x0F) - 8;
            sum += (float)lo * x[i];
            if (i + 1 < in_dim) {
                int hi = (int)((b >> 4) & 0x0F) - 8;
                sum += (float)hi * x[i + 1];
            }
        }
        y[o] += weight * sum * scale;
    }
}

static void select_top_k(const float *scores, int n, int k, int *out_ids, float *out_weights) {
    for (int i = 0; i < k; i++) {
        out_ids[i] = -1;
        out_weights[i] = -1e30f;
    }

    for (int e = 0; e < n; e++) {
        float s = scores[e];
        for (int i = 0; i < k; i++) {
            if (s > out_weights[i]) {
                for (int j = k - 1; j > i; j--) {
                    out_ids[j] = out_ids[j - 1];
                    out_weights[j] = out_weights[j - 1];
                }
                out_ids[i] = e;
                out_weights[i] = s;
                break;
            }
        }
    }

    float max_w = out_weights[0];
    for (int i = 1; i < k; i++)
        if (out_weights[i] > max_w) max_w = out_weights[i];

    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        out_weights[i] = expf(out_weights[i] - max_w);
        sum += out_weights[i];
    }
    if (sum > 0.0f) {
        for (int i = 0; i < k; i++)
            out_weights[i] /= sum;
    }
}

const void *rabbit_fetch_expert(RabbitModel *m, int global_expert_id, int promote) {
    if (!m || global_expert_id < 0)
        return NULL;

    void *cached = rabbit_cache_get(m->cache, global_expert_id);
    if (cached)
        return cached;

    const void *ptr = rabbit_expert_ptr(m, global_expert_id);
    if (!ptr)
        return NULL;

    if (promote && m->cache && m->expert_size > 0) {
        void *slot = rabbit_cache_put(m->cache, global_expert_id, ptr, m->expert_size);
        if (slot)
            return slot;
    }
    return ptr;
}

const void *rabbit_fetch_expert_layer(RabbitModel *m, int layer, int local_expert_id, int promote) {
    if (!m || layer < 0 || local_expert_id < 0)
        return NULL;
    int global_id = rabbit_expert_global_id(m, layer, local_expert_id);
    return rabbit_fetch_expert(m, global_id, promote);
}

static void apply_expert(const RabbitModel *m, const void *expert_data,
                         float *hidden, int hidden_dim, float weight) {
    if (!expert_data || !hidden || hidden_dim <= 0) return;

    const uint8_t *bytes = (const uint8_t *)expert_data;
    size_t size = m->expert_size;
    int quant = (int)m->hdr.expert_quant; /* 0 = f32 */

    float *delta = (float *)calloc((size_t)hidden_dim, sizeof(float));
    if (!delta) return;

    int inter = m->hdr.expert_dim > 0 ? (int)m->hdr.expert_dim : hidden_dim * 2;
    if (inter > hidden_dim * 4) inter = hidden_dim * 2;
    if (inter < 1) inter = hidden_dim;

    size_t mat_elems = (size_t)inter * (size_t)hidden_dim;
    size_t mat_bytes_int8 = 4 + mat_elems;
    size_t mat_bytes_int4 = 4 + ((mat_elems + 1) / 2);
    size_t one_mat = (quant == 4) ? mat_bytes_int4 : mat_bytes_int8;
    size_t need3 = one_mat * 3;

    if (size >= need3 && inter > 0) {
        float *tmp_gate = (float *)calloc((size_t)inter, sizeof(float));
        float *tmp_up   = (float *)calloc((size_t)inter, sizeof(float));
        float *tmp_mid  = (float *)calloc((size_t)inter, sizeof(float));
        if (tmp_gate && tmp_up && tmp_mid) {
            const uint8_t *p = bytes;
            for (int part = 0; part < 3; part++) {
                float scale = 1.0f / 127.0f;
                if (size - (p - bytes) >= 4) {
                    memcpy(&scale, p, 4);
                    if (!(scale > 0.0f) || !isfinite(scale))
                        scale = 1.0f / 127.0f;
                    p += 4;
                }
                const uint8_t *W = p;
                size_t payload = (quant == 4) ? ((mat_elems + 1) / 2) : mat_elems;
                p += payload;

                if (part == 0) {
                    if (quant == 4)
                        gemv_int4(W, scale, hidden, tmp_gate, inter, hidden_dim, 1.0f);
                    else
                        gemv_int8((const int8_t *)W, scale, hidden, tmp_gate, inter, hidden_dim, 1.0f);
                } else if (part == 1) {
                    if (quant == 4)
                        gemv_int4(W, scale, hidden, tmp_up, inter, hidden_dim, 1.0f);
                    else
                        gemv_int8((const int8_t *)W, scale, hidden, tmp_up, inter, hidden_dim, 1.0f);
                } else {
                    for (int i = 0; i < inter; i++) {
                        float g = tmp_gate[i];
                        float sig = 1.0f / (1.0f + expf(-g));
                        tmp_mid[i] = (g * sig) * tmp_up[i];
                    }
                    if (quant == 4)
                        gemv_int4(W, scale, tmp_mid, delta, hidden_dim, inter, weight);
                    else
                        gemv_int8((const int8_t *)W, scale, tmp_mid, delta, hidden_dim, inter, weight);
                }
            }
            for (int i = 0; i < hidden_dim; i++)
                hidden[i] += delta[i];
            free(tmp_gate); free(tmp_up); free(tmp_mid);
            free(delta);
            return;
        }
        free(tmp_gate); free(tmp_up); free(tmp_mid);
    }

    float scale = 1.0f / 127.0f;
    const uint8_t *W = bytes;
    size_t payload = size;

    if (size >= 4 + (size_t)hidden_dim) {
        memcpy(&scale, bytes, sizeof(float));
        if (!(scale > 0.0f) || !isfinite(scale))
            scale = 1.0f / 127.0f;
        W = bytes + 4;
        payload = size - 4;
    }

    float w = weight * 0.05f;

    if (quant == 4) {
        size_t need = (size_t)hidden_dim * ((size_t)(hidden_dim + 1) / 2);
        if (payload >= need)
            gemv_int4(W, scale, hidden, delta, hidden_dim, hidden_dim, w);
    } else {
        size_t need = (size_t)hidden_dim * (size_t)hidden_dim;
        if (payload >= need)
            gemv_int8((const int8_t *)W, scale, hidden, delta, hidden_dim, hidden_dim, w);
        else if (size >= need)
            gemv_int8((const int8_t *)bytes, 1.0f / 127.0f, hidden, delta,
                      hidden_dim, hidden_dim, w);
        else {
            int usable = (int)(size < (size_t)hidden_dim ? size : (size_t)hidden_dim);
            for (int i = 0; i < usable; i++) {
                float v = ((float)bytes[i] - 128.0f) / 128.0f;
                delta[i] = weight * v * 0.1f;
            }
        }
    }

    for (int i = 0; i < hidden_dim; i++)
        hidden[i] += delta[i];

    free(delta);
}

int rabbit_moe_step(RabbitModel *m, float *hidden, int hidden_dim) {
    if (!m || !hidden || hidden_dim <= 0) return -1;
    if (m->num_experts <= 0) return 0;

    int k = m->experts_per_token;
    if (k <= 0) k = 1;
    if (k > m->num_experts) k = m->num_experts;
    if (k > 8) k = 8;

    float *scores = (float *)malloc(sizeof(float) * (size_t)m->num_experts);
    if (!scores) return -1;

    for (int e = 0; e < m->num_experts; e++) {
        float s = 0.0f;
        for (int i = 0; i < hidden_dim && i < 32; i++)
            s += hidden[i] * (float)((e + 1) * (i + 3));
        scores[e] = s;
    }

    int ids[8];
    float weights[8];
    select_top_k(scores, m->num_experts, k, ids, weights);
    free(scores);

    /* Legacy path: treat as shared (layer 0) */
    for (int i = 0; i < k; i++) {
        if (ids[i] < 0) continue;
        const void *expert = rabbit_fetch_expert_layer(m, 0, ids[i], 1);
        if (!expert) continue;
        apply_expert(m, expert, hidden, hidden_dim, weights[i]);
    }
    return 0;
}

int rabbit_ref_forward(RabbitModel *m, const char *prompt, float *out_hidden) {
    if (!m || !prompt || !out_hidden) return -1;
    int dim = m->hidden_dim;
    if (dim <= 0) return -1;

    memset(out_hidden, 0, sizeof(float) * (size_t)dim);

    const unsigned char *p = (const unsigned char *)prompt;
    int i = 0;
    while (*p) {
        out_hidden[i % dim] += ((float)(*p % 64) - 32.0f) / 32.0f;
        p++;
        i++;
    }

    int layers = m->n_layers > 0 ? m->n_layers : 1;
    for (int L = 0; L < layers; L++) {
        if (rabbit_moe_step(m, out_hidden, dim) != 0)
            return -1;
    }
    return 0;
}
