#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <math.h>
#include <time.h>

int  rabbit_moe_step(RabbitModel *m, float *hidden, int hidden_dim);
int  rabbit_layer_step(RabbitModel *m, float *hidden, int hidden_dim);
void rabbit_attn_reset(void);

int  rabbit_tokenize(const char *text, int *out_ids, int max_ids);
int  rabbit_tokenizer_vocab(void);
void rabbit_detokenize(int id, char *out, int out_size);

const char *rabbit_version(void) {
    return RABBIT_VERSION;
}

void rabbit_print_info(const RabbitModel *m) {
    if (!m) return;
    printf("  name               %s\n", m->name);
    printf("  layers             %u\n", m->hdr.n_layers);
    printf("  hidden_dim         %u\n", m->hdr.hidden_dim);
    printf("  vocab_size         %u\n", m->hdr.vocab_size);
    printf("  experts            %u (top-%u)\n", m->hdr.n_experts, m->hdr.experts_per_token);
    printf("  dense              %.2f MB\n", m->dense_size / (1024.0 * 1024.0));
    printf("  expert record      %.2f KB\n", m->expert_size / 1024.0);
    printf("  experts total      %.2f MB\n", m->hdr.total_experts_size / (1024.0 * 1024.0));
    printf("  dense_quant        %u-bit\n", m->hdr.dense_quant ? m->hdr.dense_quant : 16);
    printf("  expert_quant       %u-bit\n", m->hdr.expert_quant ? m->hdr.expert_quant : 16);
    printf("  expert_map         %s\n", m->expert_map ? "mapped" : "unavailable");
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
            fprintf(stderr, "error: out of memory allocating dense weights (%.1f MB)\n",
                    hdr.dense_size / (1024.0 * 1024.0));
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
            } else {
                fprintf(stderr, "warning: could not memory-map expert region\n");
            }
        }
    }

    if (fd >= 0) close(fd);

    m->cache = rabbit_cache_create(4);
    if (m->cache)
        rabbit_cache_set_expert_size(m->cache, m->expert_size);

    return m;
}

void rabbit_free(RabbitModel *model) {
    if (!model) return;
    free(model->dense);
    rabbit_munmap_experts(model->expert_map, model->expert_map_size, model->expert_fd);
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
        for (int i = 1; i < n; i++) {
            if (probs[i] > best_p) {
                best_p = probs[i];
                best = i;
            }
        }
        return best;
    }

    float *tmp = (float *)malloc(sizeof(float) * (size_t)n);
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
        if (r <= cdf) {
            chosen = i;
            break;
        }
    }
    free(tmp);
    return chosen;
}

static void lm_head(const float *hidden, int hidden_dim, float *logits, int vocab) {
    for (int v = 0; v < vocab; v++) {
        float s = 0.0f;
        for (int i = 0; i < hidden_dim && i < 32; i++)
            s += hidden[i] * (float)((v * 17 + i * 3) % 97);
        logits[v] = s;
    }
}

static void generate(RabbitModel *m, const char *prompt, int max_new_tokens, float temperature) {
    int dim = m->hidden_dim > 0 ? m->hidden_dim : 64;
    int vocab = rabbit_tokenizer_vocab();
    if (vocab > 4096) vocab = 4096;

    float *hidden = (float *)calloc((size_t)dim, sizeof(float));
    float *logits = (float *)calloc((size_t)vocab, sizeof(float));
    int *tokens = (int *)malloc(sizeof(int) * ((size_t)strlen(prompt) + (size_t)max_new_tokens + 16));

    if (!hidden || !logits || !tokens) {
        free(hidden);
        free(logits);
        free(tokens);
        fprintf(stderr, "error: out of memory during generation\n");
        return;
    }

    rabbit_attn_reset();

    int n_tokens = rabbit_tokenize(prompt, tokens, 512);
    if (n_tokens <= 0) {
        printf("(empty prompt)\n\n");
        free(hidden);
        free(logits);
        free(tokens);
        return;
    }

    for (int t = 0; t < n_tokens; t++) {
        for (int i = 0; i < dim; i++)
            hidden[i] *= 0.9f;
        hidden[tokens[t] % dim] += ((float)(tokens[t] % 256) - 64.0f) / 64.0f;

        for (int layer = 0; layer < m->n_layers; layer++)
            rabbit_layer_step(m, hidden, dim);
    }

    char piece[128];
    for (int t = 0; t < max_new_tokens; t++) {
        lm_head(hidden, dim, logits, vocab);
        softmax(logits, vocab);

        int next = sample_token(logits, vocab, temperature);
        rabbit_detokenize(next, piece, sizeof(piece));
        fputs(piece, stdout);
        fflush(stdout);

        for (int i = 0; i < dim; i++)
            hidden[i] *= 0.92f;
        hidden[next % dim] += ((float)(next % 256) - 64.0f) / 64.0f;

        for (int layer = 0; layer < m->n_layers; layer++)
            rabbit_layer_step(m, hidden, dim);

        tokens[n_tokens++] = next;
    }

    printf("\n\n");
    free(hidden);
    free(logits);
    free(tokens);
}

rabbit_status rabbit_chat(RabbitModel *model) {
    return rabbit_chat_ex(model, 64, 0.85f);
}

rabbit_status rabbit_chat_ex(RabbitModel *model, int max_tokens, float temperature) {
    if (!model) return RABBIT_ERR_ARGS;

    srand((unsigned)time(NULL));

    printf("\n");
    printf("Rabbit %s\n", rabbit_version());
    printf("----------------------------------------\n");
    rabbit_print_info(model);
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

        if (!fgets(line, sizeof(line), stdin))
            break;

        size_t len = strlen(line);
        if (len && line[len - 1] == '\n')
            line[len - 1] = 0;

        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0)
            break;

        if (line[0] == 0)
            continue;

        generate(model, line, max_tokens, temperature);
    }

    return RABBIT_OK;
}
