#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <math.h>
#include <time.h>

int rabbit_moe_step(RabbitModel *m, float *hidden, int hidden_dim);

/* Tokenizer API */
int  rabbit_tokenize(const char *text, int *out_ids, int max_ids);
void rabbit_detokenize_seq(const int *ids, int n, char *out, int out_size);
int  rabbit_tokenizer_vocab(void);

const char *rabbit_version(void) {
    return RABBIT_VERSION;
}

void rabbit_print_info(const RabbitModel *m) {
    if (!m) return;
    printf("Model name          : %s\n", m->name);
    printf("Layers              : %u\n", m->hdr.n_layers);
    printf("Hidden dim          : %u\n", m->hdr.hidden_dim);
    printf("Vocab size          : %u\n", m->hdr.vocab_size);
    printf("Experts             : %u\n", m->hdr.n_experts);
    printf("Experts per token   : %u\n", m->hdr.experts_per_token);
    printf("Dense size          : %.2f MB\n", m->dense_size / (1024.0 * 1024.0));
    printf("One expert size     : %.2f KB\n", m->expert_size / 1024.0);
    printf("Total experts size  : %.2f MB\n", m->hdr.total_experts_size / (1024.0 * 1024.0));
    printf("Dense quant         : %u-bit\n", m->hdr.dense_quant ? m->hdr.dense_quant : 16);
    printf("Expert quant        : %u-bit\n", m->hdr.expert_quant ? m->hdr.expert_quant : 16);
    printf("Expert map          : %s\n", m->expert_map ? "active" : "none");
}

RabbitModel *rabbit_load(const char *path) {
    if (!path) return NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("open model");
        return NULL;
    }

    RabbitHeader hdr;
    ssize_t n = read(fd, &hdr, sizeof(hdr));
    if (n != (ssize_t)sizeof(hdr)) {
        fprintf(stderr, "[rabbit] failed to read header\n");
        close(fd);
        return NULL;
    }

    if (memcmp(hdr.magic, RABBIT_MAGIC, 8) != 0) {
        fprintf(stderr, "[rabbit] bad magic (not a Rabbit model)\n");
        close(fd);
        return NULL;
    }

    if (hdr.version != 1) {
        fprintf(stderr, "[rabbit] unsupported version: %u\n", hdr.version);
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
    m->expert_map = NULL;

    if (hdr.dense_size > 0) {
        m->dense = malloc(hdr.dense_size);
        if (!m->dense) {
            fprintf(stderr, "[rabbit] out of memory for dense weights\n");
            free(m);
            close(fd);
            return NULL;
        }
        n = read(fd, m->dense, hdr.dense_size);
        if (n != (ssize_t)hdr.dense_size) {
            fprintf(stderr, "[rabbit] failed to read dense blob\n");
            free(m->dense);
            free(m);
            close(fd);
            return NULL;
        }
    }

    off_t expert_offset = (off_t)(sizeof(RabbitHeader) + hdr.dense_size);
    struct stat st;
    if (fstat(fd, &st) == 0) {
        size_t remaining = (size_t)(st.st_size - expert_offset);
        if (remaining > 0 && hdr.n_experts > 0) {
            close(fd);
            fd = -1;

            if (rabbit_mmap_experts_region(path, expert_offset, remaining,
                                           &m->expert_map, &m->expert_fd) == 0) {
                m->expert_map_size = remaining;
                printf("[rabbit] experts memory-mapped (%.2f MB)\n",
                       remaining / (1024.0 * 1024.0));
            } else {
                fprintf(stderr, "[rabbit] warning: could not mmap experts\n");
            }
        }
    }

    if (fd >= 0) close(fd);

    m->cache = rabbit_cache_create(4);
    if (m->cache)
        rabbit_cache_set_expert_size(m->cache, m->expert_size);

    printf("[rabbit] model loaded\n");
    rabbit_print_info(m);

    return m;
}

void rabbit_free(RabbitModel *model) {
    if (!model) return;
    free(model->dense);
    rabbit_munmap_experts(model->expert_map, model->expert_map_size, model->expert_fd);
    rabbit_cache_free(model->cache);
    free(model);
}

/* ---------- generation helpers ---------- */

static void softmax(float *logits, int n) {
    float max_v = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max_v) max_v = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        logits[i] = expf(logits[i] - max_v);
        sum += logits[i];
    }
    if (sum > 0.0f) {
        for (int i = 0; i < n; i++)
            logits[i] /= sum;
    }
}

static int sample_top_p(const float *probs, int n, float temperature, float top_p) {
    /* Very simple sampling for the test model */
    if (temperature <= 0.0f) {
        /* greedy */
        int best = 0;
        float best_p = probs[0];
        for (int i = 1; i < n; i++) {
            if (probs[i] > best_p) {
                best_p = probs[i];
                best = i;
            }
        }
        return best;
    }

    /* Apply temperature by re-weighting */
    float *tmp = (float *)malloc(sizeof(float) * n);
    if (!tmp) return 0;

    for (int i = 0; i < n; i++)
        tmp[i] = powf(probs[i], 1.0f / temperature);

    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += tmp[i];
    if (sum > 0.0f) {
        for (int i = 0; i < n; i++) tmp[i] /= sum;
    }

    /* Nucleus (top-p) approximation: just sample from full distribution for now */
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    int chosen = n - 1;
    for (int i = 0; i < n; i++) {
        cdf += tmp[i];
        if (r <= cdf) {
            chosen = i;
            break;
        }
    }

    free(tmp);
    (void)top_p;
    return chosen;
}

/*
 * Tiny "lm head": turn hidden state into logits over the byte vocab.
 * For the synthetic model we just use a deterministic projection
 * derived from the hidden vector. This is enough to drive the loop.
 */
static void tiny_lm_head(const float *hidden, int hidden_dim, float *logits, int vocab) {
    for (int v = 0; v < vocab; v++) {
        float s = 0.0f;
        for (int i = 0; i < hidden_dim && i < 32; i++)
            s += hidden[i] * (float)((v * 17 + i * 3) % 97);
        logits[v] = s;
    }
}

static void generate(RabbitModel *m, const char *prompt, int max_new_tokens) {
    int dim = m->hidden_dim > 0 ? m->hidden_dim : 64;
    int vocab = m->vocab_size > 0 ? m->vocab_size : rabbit_tokenizer_vocab();
    if (vocab > 256) vocab = 256;   /* byte-level for now */

    float *hidden = (float *)calloc(dim, sizeof(float));
    float *logits = (float *)calloc(vocab, sizeof(float));
    int   *tokens = (int *)malloc(sizeof(int) * (strlen(prompt) + max_new_tokens + 8));

    if (!hidden || !logits || !tokens) {
        free(hidden); free(logits); free(tokens);
        return;
    }

    /* Encode prompt */
    int n_tokens = rabbit_tokenize(prompt, tokens, 512);
    if (n_tokens <= 0) {
        printf("Rabbit: (empty prompt)\n\n");
        free(hidden); free(logits); free(tokens);
        return;
    }

    /* Seed hidden state from prompt bytes */
    for (int i = 0; i < n_tokens && i < dim; i++)
        hidden[i % dim] += ((float)tokens[i] - 64.0f) / 64.0f;

    /* Run MoE layers on the prompt representation */
    for (int layer = 0; layer < m->n_layers; layer++)
        rabbit_moe_step(m, hidden, dim);

    printf("Rabbit: ");
    fflush(stdout);

    /* Generate new tokens */
    for (int t = 0; t < max_new_tokens; t++) {
        tiny_lm_head(hidden, dim, logits, vocab);
        softmax(logits, vocab);

        int next = sample_top_p(logits, vocab, 0.9f, 0.9f);

        /* Print printable characters, skip pure control bytes */
        if (next >= 32 && next < 127)
            putchar((char)next);
        else if (next == '\n')
            putchar('\n');
        else
            putchar('.');

        fflush(stdout);

        /* Feed token back into hidden state and run one MoE step */
        for (int i = 0; i < dim; i++)
            hidden[i] *= 0.95f;   /* light decay */
        hidden[next % dim] += ((float)next - 64.0f) / 64.0f;

        rabbit_moe_step(m, hidden, dim);

        tokens[n_tokens++] = next;
    }

    printf("\n\n");

    free(hidden);
    free(logits);
    free(tokens);
}

rabbit_status rabbit_chat(RabbitModel *model) {
    if (!model) return RABBIT_ERR_ARGS;

    srand((unsigned)time(NULL));

    printf("\n========================================\n");
    printf("  Rabbit %s  \ud83d\udc07\n", rabbit_version());
    printf("========================================\n");
    rabbit_print_info(model);
    printf("========================================\n\n");

    printf("Generation loop is active (byte-level tokenizer + sampling).\n");
    printf("Type a message (or 'quit'):\n\n");

    char line[1024];
    while (1) {
        printf("> ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin))
            break;

        size_t len = strlen(line);
        if (len && line[len-1] == '\n') line[len-1] = 0;

        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0)
            break;

        if (line[0] == 0)
            continue;

        generate(model, line, 48);   /* generate up to 48 new bytes */
    }

    return RABBIT_OK;
}
