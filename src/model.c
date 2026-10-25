#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <math.h>

int rabbit_moe_step(RabbitModel *m, float *hidden, int hidden_dim);

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

static void run_tiny_forward(RabbitModel *m, const char *prompt) {
    int dim = m->hidden_dim > 0 ? m->hidden_dim : 64;
    float *hidden = (float *)calloc(dim, sizeof(float));
    if (!hidden) return;

    size_t len = strlen(prompt);
    for (size_t i = 0; i < len && (int)i < dim; i++)
        hidden[i] = ((float)(unsigned char)prompt[i] - 64.0f) / 64.0f;

    printf("[forward] %d layers, hidden_dim=%d, experts=%d (top-%d)\n",
           m->n_layers, dim, m->num_experts, m->experts_per_token);

    for (int layer = 0; layer < m->n_layers; layer++) {
        if (rabbit_moe_step(m, hidden, dim) != 0) {
            printf("[forward] moe step failed at layer %d\n", layer);
            break;
        }
    }

    /* Simple fingerprint of final state */
    float sum = 0.0f, abs_sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += hidden[i];
        abs_sum += fabsf(hidden[i]);
    }

    printf("Rabbit: \"%s\"\n", prompt);
    printf("        forward ok  |  sum=%.4f  |  L1=%.4f\n\n", sum, abs_sum);

    free(hidden);
}

rabbit_status rabbit_chat(RabbitModel *model) {
    if (!model) return RABBIT_ERR_ARGS;

    printf("\n========================================\n");
    printf("  Rabbit %s  \ud83d\udc07\n", rabbit_version());
    printf("========================================\n");
    rabbit_print_info(model);
    printf("========================================\n\n");

    printf("Real expert matmul path is active.\n");
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

        run_tiny_forward(model, line);
    }

    return RABBIT_OK;
}
