#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <math.h>
#include <time.h>

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

    if (m->cache)
        rabbit_cache_stats(m->cache, &out->cache_hits, &out->cache_misses, &out->cache_inserts);
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
    printf("  tiers              RAM dense | RAM LRU experts | mmap disk\n");
    if (r.cache_hits + r.cache_misses > 0) {
        double rate = 100.0 * (double)r.cache_hits /
                      (double)(r.cache_hits + r.cache_misses);
        printf("  cache hit rate     %.1f%% (%llu hits / %llu misses)\n",
               rate,
               (unsigned long long)r.cache_hits,
               (unsigned long long)r.cache_misses);
    }
}

void rabbit_print_info(const RabbitModel *m) {
    if (!m) return;
    printf("  name               %s\n", m->name);
    printf("  layers             %u\n", m->hdr.n_layers);
    printf("  hidden_dim         %u\n", m->hdr.hidden_dim);
    printf("  vocab_size         %u\n", m->hdr.vocab_size);
    printf("  experts/layer      %d (top-%d)\n", m->num_experts, m->experts_per_token);
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
    uint64_t hits, misses, inserts;
    rabbit_cache_stats(m->cache, &hits, &misses, &inserts);
    uint64_t total = hits + misses;
    double rate = total ? (100.0 * (double)hits / (double)total) : 0.0;
    printf("\n----------------------------------------\n");
    printf("  cache hits         %llu\n", (unsigned long long)hits);
    printf("  cache misses       %llu\n", (unsigned long long)misses);
    printf("  cache inserts      %llu\n", (unsigned long long)inserts);
    printf("  cache hit rate     %.1f%%\n", rate);
    printf("----------------------------------------\n");
}

static int choose_cache_slots(size_t expert_size, size_t budget, size_t dense_size) {
    const char *env = getenv("RABBIT_CACHE_SLOTS");
    if (env && env[0]) {
        long n = strtol(env, NULL, 10);
        if (n >= 1 && n <= RABBIT_MAX_CACHE_SLOTS)
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

static void setup_expert_table(RabbitModel *m) {
    int epl = m->num_experts > 0 ? m->num_experts : 0;
    int layers = m->n_layers > 0 ? m->n_layers : 1;
    m->experts_per_layer = epl;

    int nrec = 0;
    if (m->expert_size > 0 && m->hdr.total_experts_size > 0)
        nrec = (int)(m->hdr.total_experts_size / m->expert_size);

    if (epl > 0 && nrec == epl * layers) {
        m->n_expert_records = nrec;
        m->experts_shared = 0;
    } else if (epl > 0 && nrec == epl) {
        m->n_expert_records = nrec;
        m->experts_shared = 1;
    } else if (nrec > 0) {
        m->n_expert_records = nrec;
        m->experts_shared = (nrec <= epl || epl == 0) ? 1 : 0;
        if (!m->experts_shared && layers > 0 && nrec % layers == 0)
            m->experts_per_layer = nrec / layers;
        else if (m->experts_shared)
            m->num_experts = nrec;
    } else {
        m->n_expert_records = epl;
        m->experts_shared = 1;
    }
}

static int is_ref_v2_model(const RabbitModel *m) {
    if (!m || !m->dense) return 0;
    if (strncmp(m->name, "ref-v2", 6) == 0) return 1;
    if (strncmp(m->name, "micro-hf", 8) == 0) return 1;
    if (m->hdr.dense_quant != 0) return 0;
    if (m->hdr.expert_dim == 0) return 0;
    int H = m->hidden_dim;
    int V = m->vocab_size;
    int L = m->n_layers;
    int E = m->num_experts;
    if (H <= 0 || V <= 0 || L <= 0) return 0;
    size_t need = (size_t)V * H + (size_t)H * V;
    for (int i = 0; i < L; i++) {
        need += (size_t)H * 2;
        need += (size_t)3 * H * H;
        need += (size_t)H * H;
        need += (size_t)E * H;
    }
    need *= sizeof(float);
    return m->dense_size >= need;
}

RabbitModel *rabbit_load(const char *path) {
    if (!path) return NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("open");
        return NULL;
    }

    RabbitHeader hdr;
    ssize_t n = read(fd, &hdr, sizeof(hdr));
    if (n != (ssize_t)sizeof(hdr)) {
        fprintf(stderr, "error: incomplete header\n");
        close(fd);
        return NULL;
    }

    if (memcmp(hdr.magic, RABBIT_MAGIC, 8) != 0) {
        fprintf(stderr, "error: invalid magic (not a Rabbit model)\n");
        close(fd);
        return NULL;
    }
    if (hdr.version != 1) {
        fprintf(stderr, "error: unsupported format version %u\n", hdr.version);
        close(fd);
        return NULL;
    }

    size_t budget = rabbit_default_mem_budget();

    if (hdr.dense_size > budget) {
        fprintf(stderr,
                "error: dense section (%.1f MB) exceeds memory budget (%.1f MB).\n"
                "       Set RABBIT_MEM_BUDGET_MB higher or use a smaller model.\n",
                hdr.dense_size / (1024.0 * 1024.0),
                budget / (1024.0 * 1024.0));
        close(fd);
        return NULL;
    }

    RabbitModel *m = calloc(1, sizeof(RabbitModel));
    if (!m) {
        close(fd);
        return NULL;
    }

    m->hdr = hdr;
    strncpy(m->name, hdr.name, sizeof(m->name) - 1);
    m->n_layers = (int)hdr.n_layers;
    m->hidden_dim = (int)hdr.hidden_dim;
    m->vocab_size = (int)hdr.vocab_size;
    m->num_experts = (int)hdr.n_experts;
    m->experts_per_token = (int)hdr.experts_per_token;
    m->expert_size = (size_t)hdr.expert_size;
    m->dense_size = (size_t)hdr.dense_size;
    m->expert_fd = -1;
    m->mem_budget = budget;
    m->cache_slots = choose_cache_slots(m->expert_size, budget, m->dense_size);
    setup_expert_table(m);

    if (hdr.dense_size > 0) {
        m->dense = malloc(hdr.dense_size);
        if (!m->dense) {
            fprintf(stderr, "error: out of memory for dense weights\n");
            free(m);
            close(fd);
            return NULL;
        }
        n = read(fd, m->dense, hdr.dense_size);
        if (n != (ssize_t)hdr.dense_size) {
            fprintf(stderr, "error: failed to read dense blob\n");
            free(m->dense);
            free(m);
            close(fd);
            return NULL;
        }
    }
    close(fd);

    void *base = NULL;
    size_t map_size = 0;
    int map_fd = -1;
    if (rabbit_mmap_file(path, &base, &map_size, &map_fd) == 0) {
        size_t expert_offset = sizeof(RabbitHeader) + (size_t)hdr.dense_size;
        if (expert_offset < map_size && m->n_expert_records > 0) {
            m->expert_map_base = base;
            m->expert_map_size = map_size;
            m->expert_fd = map_fd;
            m->expert_map = (uint8_t *)base + expert_offset;
        } else {
            rabbit_munmap_file(base, map_size, map_fd);
        }
    } else {
        fprintf(stderr, "warning: could not memory-map model file\n");
    }

    m->cache = rabbit_cache_create(m->cache_slots);
    if (m->cache)
        rabbit_cache_set_expert_size(m->cache, m->expert_size);

    return m;
}

void rabbit_free(RabbitModel *model) {
    if (!model) return;
    print_cache_stats(model);
    free(model->dense);
    rabbit_munmap_file(model->expert_map_base, model->expert_map_size, model->expert_fd);
    rabbit_cache_free(model->cache);
    rabbit_attn_reset();
    free(model);
}

static void softmax(float *logits, int n) {
    float max_v = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max_v) max_v = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        logits[i] = expf(logits[i] - max_v);
        sum += logits[i];
    }
    if (sum > 0.0f)
        for (int i = 0; i < n; i++)
            logits[i] /= sum;
}

static int sample_token(const float *probs, int n, float temperature) {
    if (temperature <= 0.0f) {
        int best = 0;
        float best_p = probs[0];
        for (int i = 1; i < n; i++)
            if (probs[i] > best_p) { best_p = probs[i]; best = i; }
        return best;
    }
    float *tmp = malloc(sizeof(float) * (size_t)n);
    if (!tmp) return 0;
    for (int i = 0; i < n; i++)
        tmp[i] = powf(probs[i], 1.0f / temperature);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += tmp[i];
    if (sum > 0.0f)
        for (int i = 0; i < n; i++) tmp[i] /= sum;
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    int chosen = n - 1;
    for (int i = 0; i < n; i++) {
        cdf += tmp[i];
        if (r <= cdf) { chosen = i; break; }
    }
    free(tmp);
    return chosen;
}

static void lm_head_toy(const float *hidden, int hidden_dim, float *logits, int vocab) {
    for (int v = 0; v < vocab; v++) {
        float s = 0.0f;
        for (int i = 0; i < hidden_dim && i < 32; i++)
            s += hidden[i] * (float)((v * 17 + i * 3) % 97);
        logits[v] = s;
    }
}

static void generate_v2(RabbitModel *m, const char *prompt, int max_new_tokens, float temperature) {
    int vocab = m->vocab_size > 0 ? m->vocab_size : 256;
    int max_ids = (int)strlen(prompt) + max_new_tokens + 16;
    int *tokens = (int *)malloc(sizeof(int) * (size_t)max_ids);
    float *hidden = (float *)malloc(sizeof(float) * (size_t)m->hidden_dim);
    float *logits = (float *)malloc(sizeof(float) * (size_t)vocab);
    if (!tokens || !hidden || !logits) {
        free(tokens); free(hidden); free(logits);
        fprintf(stderr, "error: out of memory during v2 generation\n");
        return;
    }

    int n = 0;
    for (const unsigned char *p = (const unsigned char *)prompt; *p && n < max_ids - max_new_tokens - 1; p++)
        tokens[n++] = (int)(*p % vocab);
    if (n == 0)
        tokens[n++] = 1;

    char piece[8];
    for (int t = 0; t < max_new_tokens && n < max_ids; t++) {
        if (rabbit_ref_v2_forward(m, tokens, n, hidden, logits) != 0) {
            fprintf(stderr, "error: ref-v2 forward failed\n");
            break;
        }
        if (temperature <= 0.0f) {
            int best = 0;
            float bv = logits[0];
            for (int i = 1; i < vocab; i++)
                if (logits[i] > bv) { bv = logits[i]; best = i; }
            tokens[n++] = best;
        } else {
            softmax(logits, vocab);
            tokens[n++] = sample_token(logits, vocab, temperature);
        }
        int id = tokens[n - 1];
        if (id >= 0 && id < 256) {
            piece[0] = (char)id;
            piece[1] = 0;
            fputs(piece, stdout);
        } else {
            printf("?");
        }
        fflush(stdout);
    }
    printf("\n\n");
    free(tokens); free(hidden); free(logits);
}

static void generate(RabbitModel *m, const char *prompt, int max_new_tokens, float temperature) {
    if (is_ref_v2_model(m)) {
        generate_v2(m, prompt, max_new_tokens, temperature);
        return;
    }

    int dim = m->hidden_dim > 0 ? m->hidden_dim : 64;
    int vocab = rabbit_tokenizer_vocab();
    if (vocab > 4096) vocab = 4096;

    float *hidden = calloc((size_t)dim, sizeof(float));
    float *logits = calloc((size_t)vocab, sizeof(float));
    int *tokens = malloc(sizeof(int) * ((size_t)strlen(prompt) + (size_t)max_new_tokens + 16));
    if (!hidden || !logits || !tokens) {
        free(hidden); free(logits); free(tokens);
        fprintf(stderr, "error: out of memory during generation\n");
        return;
    }

    rabbit_attn_reset();
    int n_tokens = rabbit_tokenize(prompt, tokens, 512);
    if (n_tokens <= 0) {
        printf("(empty prompt)\n\n");
        free(hidden); free(logits); free(tokens);
        return;
    }

    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < dim; i++) hidden[i] *= 0.9f;
        hidden[tokens[t] % dim] += ((float)(tokens[t] % 256) - 64.0f) / 64.0f;
        for (int layer = 0; layer < m->n_layers; layer++)
            rabbit_layer_step(m, hidden, dim);
    }

    char piece[128];
    for (int t = 0; t < max_new_tokens; t++) {
        lm_head_toy(hidden, dim, logits, vocab);
        softmax(logits, vocab);
        int next = sample_token(logits, vocab, temperature);
        rabbit_detokenize(next, piece, sizeof(piece));
        fputs(piece, stdout);
        fflush(stdout);

        for (int i = 0; i < dim; i++) hidden[i] *= 0.92f;
        hidden[next % dim] += ((float)(next % 256) - 64.0f) / 64.0f;
        for (int layer = 0; layer < m->n_layers; layer++)
            rabbit_layer_step(m, hidden, dim);
        tokens[n_tokens++] = next;
    }
    printf("\n\n");
    free(hidden); free(logits); free(tokens);
}

rabbit_status rabbit_chat(RabbitModel *model) {
    return rabbit_chat_ex(model, 64, 0.85f);
}

rabbit_status rabbit_chat_ex(RabbitModel *model, int max_tokens, float temperature) {
    if (!model) return RABBIT_ERR_ARGS;
    srand((unsigned)time(NULL));

    printf("\nRabbit %s\n", rabbit_version());
    printf("----------------------------------------\n");
    rabbit_print_info(model);
    if (is_ref_v2_model(model))
        printf("  forward             ref-v2 (faithful)\n");
    else
        printf("  forward             legacy\n");
    printf("----------------------------------------\n");
    printf("  max_tokens         %d\n", max_tokens);
    printf("  temperature        %.2f\n", temperature);
    printf("  tokenizer_vocab    %d\n", rabbit_tokenizer_vocab());
    printf("----------------------------------------\n\n");
    printf("Enter text (quit to exit).\n\n");

    char line[1024];
    while (1) {
        printf("> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        size_t len = strlen(line);
        if (len && line[len - 1] == '\n') line[len - 1] = 0;
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) break;
        if (line[0] == 0) continue;
        generate(model, line, max_tokens, temperature);
    }
    return RABBIT_OK;
}
