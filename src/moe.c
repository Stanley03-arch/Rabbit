#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * MoE expert application (non-v2 path).
 * expert_quant 0 = f32 (not used on this path for ref-v2).
 */

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
    for (int o = 0; o < out_dim; o++) {
        float sum = 0.0f;
        const uint8_t *row = W + (size_t)o * ((in_dim + 1) / 2);
        for (int i = 0; i < in_dim; i++) {
            int byte = row[i / 2];
            int nibble = (i & 1) ? (byte >> 4) : (byte & 0x0f);
            int8_t q = (int8_t)(nibble - 8);
            sum += (float)q * x[i];
        }
        y[o] += weight * sum * scale;
    }
}

void rabbit_apply_expert(const RabbitModel *m, const void *expert_data,
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
