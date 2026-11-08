#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <math.h>
#include <time.h>
#include <errno.h>

int  rabbit_layer_step(RabbitModel *m, float *hidden, int hidden_dim);
void rabbit_attn_reset(void);
int  rabbit_tokenize(const char *text, int *out_ids, int max_ids);
int  rabbit_tokenizer_vocab(void);
void rabbit_detokenize(int id, char *out, int out_size);

const char *rabbit_version(void) {
    return RABBIT_VERSION;
}

size_t rabbit_default_mem_budget(void) {
    const char *env = getenv("RABBIT_MEM_BUDGET_MB");
    if (env && env[0]) {
        long mb = strtol(env, NULL, 10);
        if (mb > 64 && mb < 65536)
            return (size_t)mb * 1024u * 1024u;
    }
    return (size_t)RABBIT_DEFAULT_MEM_BUDGET_MB * 1024u * 1024u;
}

void rabbit_memory_report(const RabbitModel *m, RabbitMemoryReport *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!m) return;
    out->dense_bytes = m->dense_size;
    out->cache_slots = m->cache_slots > 0 ? m->cache_slots : 4;
    out->cache_bytes = (size_t)out->cache_slots * m->expert_size;
    const int max_ctx = 128;
    out->kv_bytes = (size_t)2 * max_ctx * (size_t)m->hidden_dim * sizeof(float);
    out->total_estimate = out->dense_bytes + out->cache_bytes + out->kv_bytes;
    out->budget_bytes = m->mem_budget > 0 ? m->mem_budget : rabbit_default_mem_budget();
    out->over_budget = (out->total_estimate > out->budget_bytes) ? 1 : 0;
}

void rabbit_print_memory(const RabbitModel *m) {
    RabbitMemoryReport r;
    rabbit_memory_report(m, &r);
    printf("  mem budget         %.1f MB\n", r.budget_bytes / (1024.0 * 1024.0));
    printf("  dense (RAM)        %.2f MB\n", r.dense_bytes / (1024.0 * 1024.0));
    printf("  expert cache       %.2f MB (%d slots)\n",
           r.cache_bytes / (1024.0 * 1024.0), r.cache_slots);
    printf("  KV cache est.      %.2f MB\n", r.kv_bytes / (1024.0 * 1024.0));
    printf("  total estimate     %.2f MB%s\n",
           r.total_estimate / (1024.0 * 1024.0),
           r.over_budget ? "  [OVER BUDGET]" : "");
}

void rabbit_print_info(const RabbitModel *m) {
    if (!m) return;
    printf("  name               %s\n", m->name);
    printf("  layers             %u\n", m->hdr.n_layers);
    printf("  hidden_dim         %u\n", m->hdr.hidden_dim);
    printf("  vocab_size         %u\n", m->hdr.vocab_size);
    printf("  experts/layer      %d (top-%d)\n", m->num_experts, m->experts_per_token);
    {
        uint32_t n_shared = 0;
        memcpy(&n_shared, m->hdr.reserved + 12, 4);
        if (n_shared > 0)
            printf("  shared experts     %u (always-on)\n", n_shared);
    }
    printf("  expert records     %d (%s)\n",
           m->n_expert_records,
           m->experts_shared ? "shared" : "per-layer");
    printf("  dense              %.2f MB\n", m->dense_size / (1024.0 * 1024.0));
    printf("  expert record      %.2f KB\n", m->expert_size / 1024.0);
    printf("  experts total      %.2f MB\n", m->hdr.total_experts_size / (1024.0 * 1024.0));
    if (m->hdr.dense_quant)
        printf("  dense_quant        %u-bit\n", m->hdr.dense_quant);
    else
        printf("  dense_quant        f32\n");
    if (m->hdr.expert_quant)
        printf("  expert_quant       %u-bit\n", m->hdr.expert_quant);
    else
        printf("  expert_quant       f32\n");
    printf("  expert_map         %s\n", m->expert_map ? "mapped" : "unavailable");
    rabbit_print_memory(m);
}

static void print_cache_stats(const RabbitModel *m) {
    if (!m || !m->cache) return;
    uint64_t hits, misses, inserts, evictions, bytes_copied;
    int n_slots = 0, n_pinned = 0;
    rabbit_cache_stats_ex(m->cache, &hits, &misses, &inserts, &evictions, &bytes_copied,
                          &n_slots, &n_pinned);
    uint64_t total = hits + misses;
    double rate = total ? (100.0 * (double)hits / (double)total) : 0.0;
    printf("\n----------------------------------------\n");
    printf("  cache hits         %llu\n", (unsigned long long)hits);
    printf("  cache misses       %llu\n", (unsigned long long)misses);
    printf("  cache inserts      %llu\n", (unsigned long long)inserts);
    printf("  cache evictions    %llu\n", (unsigned long long)evictions);
    printf("  cache bytes copied %llu\n", (unsigned long long)bytes_copied);
    printf("  cache slots        %d (%d pinned)\n", n_slots, n_pinned);
    printf("  cache hit rate     %.1f%%\n", rate);
    printf("----------------------------------------\n");
}

static int choose_cache_slots(size_t expert_size, size_t budget, size_t dense_size) {
    const char *env_slots = getenv("RABBIT_CACHE_SLOTS");
    if (env_slots && env_slots[0]) {
        long n = strtol(env_slots, NULL, 10);
        if (n < 1) n = 1;
        if (n > RABBIT_MAX_CACHE_SLOTS) n = RABBIT_MAX_CACHE_SLOTS;
        return (int)n;
    }
    if (expert_size == 0) return 2;
    size_t remain = 0;
    if (budget > dense_size + 8 * 1024 * 1024)
        remain = budget - dense_size - 8 * 1024 * 1024;
    int slots = (int)(remain / expert_size);
    if (slots < 1) slots = 1;
    if (slots > RABBIT_MAX_CACHE_SLOTS) slots = RABBIT_MAX_CACHE_SLOTS;
    return slots;
}

static void pin_shared_experts(RabbitModel *m) {
    if (!m || !m->cache || !m->expert_map) return;
    uint32_t n_shared = 0;
    memcpy(&n_shared, m->hdr.reserved + 12, 4);
    if (n_shared == 0 || n_shared > 64) return;
    int epl = m->experts_per_layer > 0 ? m->experts_per_layer : m->num_experts;
    int routed = m->num_experts > 0 ? m->num_experts : 0;
    int pinned = 0;
    for (int layer = 0; layer < m->n_layers; layer++) {
        for (uint32_t s = 0; s < n_shared; s++) {
            int local = routed + (int)s;
            if (local >= epl) break;
            int gid = rabbit_expert_global_id(m, layer, local);
            const void *ptr = rabbit_expert_ptr(m, gid);
            if (ptr && rabbit_cache_pin(m->cache, gid, ptr) == 0)
                pinned++;
        }
    }
    if (pinned > 0)
        fprintf(stderr, "rabbit: pinned %d shared-expert slot(s) in cache\n", pinned);
}

/* REMAINDER of model.c (load/free/generate/chat) continues in repository;
 * this truncated push is incomplete — DO NOT USE as sole source.
 * Full 20KB local copy must be re-applied.
 */
