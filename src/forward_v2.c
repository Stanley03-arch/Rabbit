#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Faithful rabbit-ref-v2 forward — must match tools/ref_forward_v2.py
 *
 * Experts on disk: either shared (E records) or per-layer (L*E records
 * ordered L0E0..L0E{n-1}, L1E0, ...).
 * expert_quant 0 = pure f32 gate/up/down; 8 = scale+int8 each.
 */

#define V2_EPS 1e-5f
#define V2_MAX_CTX 256
#define V2_MAX_HEADS 16

typedef struct {
    const float *embed;
    const float *lm_head;
    int n_layers;
    int hidden;
    int vocab;
    int n_experts;
    int top_k;
    int inter;
    int n_heads;
    int head_dim;

    const float *rms_attn[8];
    const float *rms_ffn[8];
    const float *w_qkv[8];
    const float *w_o[8];
    const float *w_router[8];
} RefV2Weights;

typedef struct {
    float *k;
    float *v;
    int len;
    int max_len;
} RefV2KV;

static void rmsnorm(const float *x, const float *w, float *out, int n) {
    float var = 0.0f;
    for (int i = 0; i < n; i++)
        var += x[i] * x[i];
    var /= (float)n;
    float inv = 1.0f / sqrtf(var + V2_EPS);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * inv * w[i];
}

static void gemv_f32(const float *W, const float *x, float *y, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = 0.0f;
        const float *row = W + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; i++)
            sum += row[i] * x[i];
        y[o] = sum;
    }
}

static void gemv_int8_row(const int8_t *W, float scale, const float *x, float *y,
                          int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float sum = 0.0f;
        const int8_t *row = W + (size_t)o * in_dim;
        for (int i = 0; i < in_dim; i++)
            sum += (float)row[i] * x[i];
        y[o] = sum * scale;
    }
}

static void apply_rope_head(float *x, int head_dim, int pos) {
    int half = head_dim / 2;
    if (half < 1) return;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(10000.0f, (float)i / (float)half);
        float angle = (float)pos * freq;
        float c = cosf(angle), s = sinf(angle);
        float x0 = x[2 * i];
        float x1 = x[2 * i + 1];
        x[2 * i]     = x0 * c - x1 * s;
        x[2 * i + 1] = x0 * s + x1 * c;
    }
}

static int parse_weights(const RabbitModel *m, RefV2Weights *w) {
    if (!m || !m->dense || m->dense_size < 64) return -1;

    w->n_layers = m->n_layers;
    w->hidden = m->hidden_dim;
    w->vocab = m->vocab_size;
    w->n_experts = m->num_experts;
    w->top_k = m->experts_per_token > 0 ? m->experts_per_token : 2;
    w->inter = m->hdr.expert_dim > 0 ? (int)m->hdr.expert_dim : m->hidden_dim * 2;
    w->n_heads = 4;
    if (w->hidden % w->n_heads != 0)
        w->n_heads = 1;
    w->head_dim = w->hidden / w->n_heads;

    if (w->n_layers > 8) return -1;

    const float *p = (const float *)m->dense;
    size_t floats = m->dense_size / sizeof(float);
    size_t need = (size_t)w->vocab * w->hidden + (size_t)w->hidden * w->vocab;
    for (int L = 0; L < w->n_layers; L++) {
        need += (size_t)w->hidden * 2;
        need += (size_t)3 * w->hidden * w->hidden;
        need += (size_t)w->hidden * w->hidden;
        need += (size_t)w->n_experts * w->hidden;
    }
    if (need > floats) return -1;

    w->embed = p;
    p += (size_t)w->vocab * w->hidden;
    w->lm_head = p;
    p += (size_t)w->hidden * w->vocab;

    for (int L = 0; L < w->n_layers; L++) {
        w->rms_attn[L] = p; p += w->hidden;
        w->rms_ffn[L]  = p; p += w->hidden;
        w->w_qkv[L]    = p; p += (size_t)3 * w->hidden * w->hidden;
        w->w_o[L]      = p; p += (size_t)w->hidden * w->hidden;
        w->w_router[L] = p; p += (size_t)w->n_experts * w->hidden;
    }
    return 0;
}

static void attention_step(const RefV2Weights *w, int layer, float *h, RefV2KV *kv, int pos) {
    int H = w->hidden;
    int nh = w->n_heads;
    int hd = w->head_dim;

    float *xn = (float *)malloc(sizeof(float) * (size_t)H);
    float *qkv = (float *)malloc(sizeof(float) * (size_t)3 * H);
    float *q = (float *)malloc(sizeof(float) * (size_t)H);
    float *k = (float *)malloc(sizeof(float) * (size_t)H);
    float *v = (float *)malloc(sizeof(float) * (size_t)H);
    float *att_out = (float *)calloc((size_t)H, sizeof(float));
    float *proj = (float *)malloc(sizeof(float) * (size_t)H);
    if (!xn || !qkv || !q || !k || !v || !att_out || !proj) {
        free(xn); free(qkv); free(q); free(k); free(v); free(att_out); free(proj);
        return;
    }

    rmsnorm(h, w->rms_attn[layer], xn, H);
    gemv_f32(w->w_qkv[layer], xn, qkv, 3 * H, H);
    memcpy(q, qkv, sizeof(float) * (size_t)H);
    memcpy(k, qkv + H, sizeof(float) * (size_t)H);
    memcpy(v, qkv + 2 * H, sizeof(float) * (size_t)H);

    for (int head = 0; head < nh; head++) {
        apply_rope_head(q + head * hd, hd, pos);
        apply_rope_head(k + head * hd, hd, pos);
    }

    if (kv->len < kv->max_len) {
        memcpy(kv->k + (size_t)kv->len * H, k, sizeof(float) * (size_t)H);
        memcpy(kv->v + (size_t)kv->len * H, v, sizeof(float) * (size_t)H);
        kv->len++;
    }

    float scale = 1.0f / sqrtf((float)hd);
    float scores[V2_MAX_CTX];

    for (int head = 0; head < nh; head++) {
        int off = head * hd;
        for (int t = 0; t < kv->len; t++) {
            float dot = 0.0f;
            const float *kt = kv->k + (size_t)t * H + off;
            for (int i = 0; i < hd; i++)
                dot += q[off + i] * kt[i];
            scores[t] = dot * scale;
        }
        float max_s = scores[0];
        for (int t = 1; t < kv->len; t++)
            if (scores[t] > max_s) max_s = scores[t];
        float sum = 0.0f;
        for (int t = 0; t < kv->len; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum += scores[t];
        }
        if (sum > 0.0f)
            for (int t = 0; t < kv->len; t++)
                scores[t] /= sum;

        for (int t = 0; t < kv->len; t++) {
            const float *vt = kv->v + (size_t)t * H + off;
            for (int i = 0; i < hd; i++)
                att_out[off + i] += scores[t] * vt[i];
        }
    }

    gemv_f32(w->w_o[layer], att_out, proj, H, H);
    for (int i = 0; i < H; i++)
        h[i] += proj[i];

    free(xn); free(qkv); free(q); free(k); free(v); free(att_out); free(proj);
}

static void select_top_k(const float *scores, int n, int k, int *ids, float *weights) {
    for (int i = 0; i < k; i++) {
        ids[i] = -1;
        weights[i] = -1e30f;
    }
    for (int e = 0; e < n; e++) {
        float s = scores[e];
        for (int i = 0; i < k; i++) {
            if (s > weights[i]) {
                for (int j = k - 1; j > i; j--) {
                    ids[j] = ids[j - 1];
                    weights[j] = weights[j - 1];
                }
                ids[i] = e;
                weights[i] = s;
                break;
            }
        }
    }
    float max_w = weights[0];
    for (int i = 1; i < k; i++)
        if (weights[i] > max_w) max_w = weights[i];
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        weights[i] = expf(weights[i] - max_w);
        sum += weights[i];
    }
    if (sum > 0.0f)
        for (int i = 0; i < k; i++)
            weights[i] /= sum;
}

static void expert_swiglu(const RabbitModel *m, int layer, int local_expert_id,
                          const float *xn, float *delta, int H, int inter, float weight) {
    const uint8_t *bytes = (const uint8_t *)rabbit_expert_ptr_layer(m, layer, local_expert_id);
    if (!bytes) return;

    int quant = m->hdr.expert_quant ? (int)m->hdr.expert_quant : 8;
    size_t mat = (size_t)inter * (size_t)H;

    float *gate = (float *)calloc((size_t)inter, sizeof(float));
    float *up = (float *)calloc((size_t)inter, sizeof(float));
    float *mid = (float *)calloc((size_t)inter, sizeof(float));
    float *down_out = (float *)calloc((size_t)H, sizeof(float));
    if (!gate || !up || !mid || !down_out) {
        free(gate); free(up); free(mid); free(down_out);
        return;
    }

    if (quant == 0) {
        /* pure f32: gate[I,H] + up[I,H] + down[H,I] */
        size_t need = (mat * 2 + (size_t)H * (size_t)inter) * sizeof(float);
        if (m->expert_size < need) {
            free(gate); free(up); free(mid); free(down_out);
            return;
        }
        const float *p = (const float *)bytes;
        gemv_f32(p, xn, gate, inter, H);
        p += mat;
        gemv_f32(p, xn, up, inter, H);
        p += mat;
        for (int i = 0; i < inter; i++) {
            float g = gate[i];
            float sig = 1.0f / (1.0f + expf(-g));
            mid[i] = (g * sig) * up[i];
        }
        gemv_f32(p, mid, down_out, H, inter);
    } else {
        size_t one = 4 + mat;
        if (m->expert_size < one * 3) {
            free(gate); free(up); free(mid); free(down_out);
            return;
        }
        const uint8_t *p = bytes;
        float scale_g, scale_u, scale_d;
        memcpy(&scale_g, p, 4); p += 4;
        gemv_int8_row((const int8_t *)p, scale_g, xn, gate, inter, H);
        p += mat;

        memcpy(&scale_u, p, 4); p += 4;
        gemv_int8_row((const int8_t *)p, scale_u, xn, up, inter, H);
        p += mat;

        for (int i = 0; i < inter; i++) {
            float g = gate[i];
            float sig = 1.0f / (1.0f + expf(-g));
            mid[i] = (g * sig) * up[i];
        }

        memcpy(&scale_d, p, 4); p += 4;
        gemv_int8_row((const int8_t *)p, scale_d, mid, down_out, H, inter);
    }

    for (int i = 0; i < H; i++)
        delta[i] += weight * down_out[i];

    free(gate); free(up); free(mid); free(down_out);
}

static void moe_step(const RabbitModel *m, const RefV2Weights *w, int layer, float *h) {
    int H = w->hidden;
    int E = w->n_experts;
    int k = w->top_k;
    if (k > E) k = E;
    if (k > 8) k = 8;

    float *xn = (float *)malloc(sizeof(float) * (size_t)H);
    float *scores = (float *)malloc(sizeof(float) * (size_t)E);
    float *delta = (float *)calloc((size_t)H, sizeof(float));
    if (!xn || !scores || !delta) {
        free(xn); free(scores); free(delta);
        return;
    }

    rmsnorm(h, w->rms_ffn[layer], xn, H);
    gemv_f32(w->w_router[layer], xn, scores, E, H);

    int ids[8];
    float weights[8];
    select_top_k(scores, E, k, ids, weights);

    for (int i = 0; i < k; i++) {
        if (ids[i] < 0) continue;
        expert_swiglu(m, layer, ids[i], xn, delta, H, w->inter, weights[i]);
    }

    for (int i = 0; i < H; i++)
        h[i] += delta[i];

    free(xn); free(scores); free(delta);
}

static void logits_from_hidden(const RefV2Weights *w, const float *h, float *logits) {
    int H = w->hidden;
    int V = w->vocab;
    for (int v = 0; v < V; v++) {
        float s = 0.0f;
        for (int i = 0; i < H; i++)
            s += w->lm_head[(size_t)i * V + v] * h[i];
        logits[v] = s;
    }
}

static int argmax_f(const float *x, int n) {
    int best = 0;
    float bv = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > bv) { bv = x[i]; best = i; }
    return best;
}

int rabbit_ref_v2_forward(RabbitModel *m, const int *token_ids, int n_tokens,
                          float *out_hidden, float *out_logits) {
    if (!m || !token_ids || n_tokens <= 0 || !out_hidden) return -1;

    RefV2Weights w;
    memset(&w, 0, sizeof(w));
    if (parse_weights(m, &w) != 0) {
        fprintf(stderr, "error: dense layout is not rabbit-ref-v2\n");
        return -1;
    }

    int H = w.hidden;
    RefV2KV kv[8];
    memset(kv, 0, sizeof(kv));
    for (int L = 0; L < w.n_layers; L++) {
        kv[L].max_len = V2_MAX_CTX;
        kv[L].k = (float *)calloc((size_t)V2_MAX_CTX * H, sizeof(float));
        kv[L].v = (float *)calloc((size_t)V2_MAX_CTX * H, sizeof(float));
        if (!kv[L].k || !kv[L].v) {
            for (int j = 0; j <= L; j++) {
                free(kv[j].k); free(kv[j].v);
            }
            return -1;
        }
    }

    float *h = (float *)calloc((size_t)H, sizeof(float));
    if (!h) {
        for (int L = 0; L < w.n_layers; L++) {
            free(kv[L].k); free(kv[L].v);
        }
        return -1;
    }

    for (int pos = 0; pos < n_tokens; pos++) {
        int tid = token_ids[pos] % w.vocab;
        if (tid < 0) tid = 0;
        memcpy(h, w.embed + (size_t)tid * H, sizeof(float) * (size_t)H);

        for (int L = 0; L < w.n_layers; L++) {
            attention_step(&w, L, h, &kv[L], pos);
            moe_step(m, &w, L, h);
        }
    }

    memcpy(out_hidden, h, sizeof(float) * (size_t)H);
    if (out_logits)
        logits_from_hidden(&w, h, out_logits);

    free(h);
    for (int L = 0; L < w.n_layers; L++) {
        free(kv[L].k);
        free(kv[L].v);
    }
    return 0;
}

int rabbit_ref_v2_greedy(RabbitModel *m, const char *prompt, int n_new,
                         int *out_ids, int max_ids) {
    if (!m || !prompt || !out_ids || max_ids < 1) return -1;

    RefV2Weights w;
    if (parse_weights(m, &w) != 0) return -1;

    int n = 0;
    for (const unsigned char *p = (const unsigned char *)prompt; *p && n < max_ids - n_new - 1; p++)
        out_ids[n++] = (int)(*p % w.vocab);
    if (n == 0)
        out_ids[n++] = 1;

    float *hidden = (float *)malloc(sizeof(float) * (size_t)w.hidden);
    float *logits = (float *)malloc(sizeof(float) * (size_t)w.vocab);
    if (!hidden || !logits) {
        free(hidden); free(logits);
        return -1;
    }

    for (int t = 0; t < n_new && n < max_ids; t++) {
        if (rabbit_ref_v2_forward(m, out_ids, n, hidden, logits) != 0) {
            free(hidden); free(logits);
            return -1;
        }
        out_ids[n++] = argmax_f(logits, w.vocab);
    }

    free(hidden);
    free(logits);
    return n;
}
