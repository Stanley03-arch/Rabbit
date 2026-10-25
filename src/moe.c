#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Mixture-of-Experts core for Rabbit.
 *
 * Design:
 *   1. Router produces scores for all experts
 *   2. Select top-k experts
 *   3. For each selected expert:
 *        - try RAM cache first
 *        - on miss, take pointer from mmap (zero-copy)
 *        - optionally promote into the tiny LRU cache
 *   4. Apply expert (for the tiny test model we use a simple linear transform)
 */

/* Simple top-k selection (k is small) */
static void select_top_k(const float *scores, int n, int k, int *out_ids, float *out_weights) {
    /* Naive but fine for tiny models / first version */
    for (int i = 0; i < k; i++) {
        out_ids[i] = -1;
        out_weights[i] = -1e30f;
    }

    for (int e = 0; e < n; e++) {
        float s = scores[e];
        for (int i = 0; i < k; i++) {
            if (s > out_weights[i]) {
                /* shift down */
                for (int j = k - 1; j > i; j--) {
                    out_ids[j] = out_ids[j-1];
                    out_weights[j] = out_weights[j-1];
                }
                out_ids[i] = e;
                out_weights[i] = s;
                break;
            }
        }
    }

    /* Softmax over the selected weights */
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

/*
 * Fetch one expert:
 *   1. Check RAM cache
 *   2. Fall back to mmap pointer (zero-copy)
 *   3. Optionally promote into cache
 */
const void *rabbit_fetch_expert(RabbitModel *m, int expert_id, int promote) {
    if (!m || expert_id < 0 || expert_id >= m->num_experts)
        return NULL;

    /* 1. Try cache */
    void *cached = rabbit_cache_get(m->cache, expert_id);
    if (cached)
        return cached;

    /* 2. Zero-copy from mmap */
    const void *ptr = rabbit_expert_ptr(m, expert_id);
    if (!ptr)
        return NULL;

    /* 3. Optionally promote into the tiny RAM cache */
    if (promote && m->cache) {
        void *ram = rabbit_cache_put(m->cache, expert_id, ptr);
        if (ram)
            return ram;
    }

    return ptr;
}

/*
 * Very simple expert application for the synthetic test model.
 * Treats the expert blob as a bias vector of length hidden_dim
 * and just adds a scaled version of it.
 *
 * This is only for proving the plumbing works.
 */
static void apply_expert_stub(const void *expert_data, size_t expert_size,
                              float *hidden, int hidden_dim, float weight) {
    if (!expert_data || !hidden || hidden_dim <= 0) return;

    const uint8_t *bytes = (const uint8_t *)expert_data;
    int usable = (int)(expert_size < (size_t)hidden_dim ? expert_size : (size_t)hidden_dim);

    for (int i = 0; i < usable; i++) {
        /* Map byte to a small float contribution */
        float v = ((float)bytes[i] - 128.0f) / 128.0f;
        hidden[i] += weight * v * 0.1f;
    }
}

/*
 * Minimal MoE step used by the tiny test model.
 * Returns 0 on success.
 */
int rabbit_moe_step(RabbitModel *m, float *hidden, int hidden_dim) {
    if (!m || !hidden || hidden_dim <= 0) return -1;
    if (m->num_experts <= 0) return 0;

    int k = m->experts_per_token;
    if (k <= 0) k = 1;
    if (k > m->num_experts) k = m->num_experts;
    if (k > 8) k = 8;   /* safety */

    /* Fake router scores (deterministic from hidden state for testing) */
    float *scores = (float *)malloc(sizeof(float) * m->num_experts);
    if (!scores) return -1;

    for (int e = 0; e < m->num_experts; e++) {
        float s = 0.0f;
        for (int i = 0; i < hidden_dim && i < 16; i++)
            s += hidden[i] * (float)((e + 1) * (i + 3));
        scores[e] = s;
    }

    int ids[8];
    float weights[8];
    select_top_k(scores, m->num_experts, k, ids, weights);
    free(scores);

    /* Apply selected experts */
    for (int i = 0; i < k; i++) {
        if (ids[i] < 0) continue;

        const void *expert = rabbit_fetch_expert(m, ids[i], 1 /* promote */);
        if (!expert) continue;

        apply_expert_stub(expert, m->expert_size, hidden, hidden_dim, weights[i]);
    }

    return 0;
}

/* Public stub kept for compatibility */
void rabbit_moe_forward_stub(RabbitModel *model) {
    (void)model;
    printf("[moe] use rabbit_moe_step() for the real path.\n");
}
