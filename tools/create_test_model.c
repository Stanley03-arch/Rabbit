/*
 * create_test_model.c
 *
 * Builds a tiny synthetic Rabbit model so we can test the loader
 * and memory mapping without needing real multi-GB weights.
 *
 * Usage:
 *   gcc -O2 -o create_test_model tools/create_test_model.c
 *   ./create_test_model models/tiny-test.rabbit
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

    /* Tiny synthetic model */
    const uint32_t n_layers = 4;
    const uint32_t hidden_dim = 64;
    const uint32_t vocab_size = 128;
    const uint32_t n_experts = 8;
    const uint32_t experts_per_token = 2;
    const uint32_t expert_dim = 128;

    /* Fake sizes (small enough for any machine) */
    const uint64_t dense_size = 64 * 1024;          /* 64 KB dense */
    const uint64_t expert_size = 16 * 1024;         /* 16 KB per expert */
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
    hdr.expert_quant = 16;

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

    /* Write dummy dense data */
    uint8_t *dense = calloc(1, dense_size);
    if (!dense) {
        close(fd);
        return 1;
    }
    /* Fill with a simple pattern */
    for (size_t i = 0; i < dense_size; i++)
        dense[i] = (uint8_t)(i & 0xFF);

    if (write(fd, dense, dense_size) != (ssize_t)dense_size) {
        perror("write dense");
        free(dense);
        close(fd);
        return 1;
    }
    free(dense);

    /* Write dummy experts */
    uint8_t *expert = calloc(1, expert_size);
    if (!expert) {
        close(fd);
        return 1;
    }

    for (uint32_t e = 0; e < n_experts; e++) {
        for (size_t i = 0; i < expert_size; i++)
            expert[i] = (uint8_t)((e * 17 + i) & 0xFF);

        if (write(fd, expert, expert_size) != (ssize_t)expert_size) {
            perror("write expert");
            free(expert);
            close(fd);
            return 1;
        }
    }
    free(expert);
    close(fd);

    printf("Created tiny test model: %s\n", out_path);
    printf("  dense   : %.1f KB\n", dense_size / 1024.0);
    printf("  experts : %u x %.1f KB\n", n_experts, expert_size / 1024.0);
    printf("  total   : %.1f KB\n", (sizeof(hdr) + dense_size + total_experts) / 1024.0);

    return 0;
}
