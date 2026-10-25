/*
 * create_test_model.c
 *
 * Builds a synthetic Rabbit model large enough to exercise
 * the real expert matmul path (int8 GEMV).
 *
 * Usage:
 *   make test-model
 *   ./rabbit chat --model models/tiny-test.rabbit
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>

#pragma pack(push, 1)
typedef struct {
    char     magic[8];
    uint32_t version;
    uint32_t header_size;

    uint32_t n_layers;
    uint32_t hidden_dim;
    uint32_t vocab_size;
    uint32_t n_experts;
    uint32_t experts_per_token;
    uint32_t expert_dim;

    uint64_t dense_size;
    uint64_t expert_size;
    uint64_t total_experts_size;

    uint32_t dense_quant;
    uint32_t expert_quant;

    char     name[64];
    uint8_t  reserved[128];
} RabbitHeader;
#pragma pack(pop)

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <output.rabbit>\n", argv[0]);
        return 1;
    }

    const char *out_path = argv[1];

    /* Dimensions chosen so each expert holds a full int8 matrix */
    const uint32_t n_layers = 4;
    const uint32_t hidden_dim = 64;
    const uint32_t vocab_size = 256;
    const uint32_t n_experts = 8;
    const uint32_t experts_per_token = 2;
    const uint32_t expert_dim = 64;

    /* Dense: small fake blob */
    const uint64_t dense_size = 128 * 1024;   /* 128 KB */

    /* Each expert = full int8 matrix [hidden_dim x hidden_dim] */
    const uint64_t expert_size = (uint64_t)hidden_dim * hidden_dim;  /* 4096 bytes */
    const uint64_t total_experts = n_experts * expert_size;

    RabbitHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "RABBIT01", 8);
    hdr.version = 1;
    hdr.header_size = sizeof(hdr);

    hdr.n_layers = n_layers;
    hdr.hidden_dim = hidden_dim;
    hdr.vocab_size = vocab_size;
    hdr.n_experts = n_experts;
    hdr.experts_per_token = experts_per_token;
    hdr.expert_dim = expert_dim;

    hdr.dense_size = dense_size;
    hdr.expert_size = expert_size;
    hdr.total_experts_size = total_experts;

    hdr.dense_quant = 16;
    hdr.expert_quant = 8;   /* int8 experts */

    strncpy(hdr.name, "tiny-test", sizeof(hdr.name) - 1);

    int fd = open(out_path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    if (write(fd, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)) {
        perror("write header");
        close(fd);
        return 1;
    }

    /* Dummy dense data */
    uint8_t *dense = calloc(1, dense_size);
    if (!dense) { close(fd); return 1; }
    for (size_t i = 0; i < dense_size; i++)
        dense[i] = (uint8_t)(i & 0xFF);
    if (write(fd, dense, dense_size) != (ssize_t)dense_size) {
        perror("write dense");
        free(dense);
        close(fd);
        return 1;
    }
    free(dense);

    /* Create int8 expert matrices with a simple pattern */
    int8_t *expert = calloc(1, expert_size);
    if (!expert) { close(fd); return 1; }

    for (uint32_t e = 0; e < n_experts; e++) {
        for (uint32_t o = 0; o < hidden_dim; o++) {
            for (uint32_t i = 0; i < hidden_dim; i++) {
                /* Deterministic but different per expert */
                int v = (int)((e * 13 + o * 7 + i * 3) % 255) - 128;
                expert[o * hidden_dim + i] = (int8_t)v;
            }
        }

        if (write(fd, expert, expert_size) != (ssize_t)expert_size) {
            perror("write expert");
            free(expert);
            close(fd);
            return 1;
        }
    }
    free(expert);
    close(fd);

    printf("Created test model: %s\n", out_path);
    printf("  layers       : %u\n", n_layers);
    printf("  hidden_dim   : %u\n", hidden_dim);
    printf("  experts      : %u (top-%u)\n", n_experts, experts_per_token);
    printf("  dense        : %.1f KB\n", dense_size / 1024.0);
    printf("  one expert   : %llu bytes (int8 matrix)\n", (unsigned long long)expert_size);
    printf("  total experts: %.1f KB\n", total_experts / 1024.0);

    return 0;
}
