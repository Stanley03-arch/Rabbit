#ifndef RABBIT_H
#define RABBIT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* ============================================================
 * Rabbit 🐇 — Core public API
 * Tiny engine for large MoE models on constrained hardware.
 * ============================================================ */

#define RABBIT_VERSION "0.1.0-dev"

/* Maximum number of experts we keep hot in RAM (keep tiny for 4GB) */
#define RABBIT_MAX_CACHE_SLOTS 8

/* Error codes */
typedef enum {
    RABBIT_OK = 0,
    RABBIT_ERR_IO,
    RABBIT_ERR_MEM,
    RABBIT_ERR_FORMAT,
    RABBIT_ERR_UNSUPPORTED,
    RABBIT_ERR_ARGS
} rabbit_status;

/* Forward declarations */
typedef struct RabbitModel RabbitModel;
typedef struct ExpertCache ExpertCache;

/* ------------------------------------------------------------
 * Model handle
 * ------------------------------------------------------------ */
struct RabbitModel {
    /* Always-resident dense / shared part */
    void   *dense;
    size_t  dense_size;

    /* Expert storage (memory-mapped file) */
    int     expert_fd;
    void   *expert_map;
    size_t  expert_map_size;
    size_t  expert_size;          /* size of one expert in bytes */
    int     num_experts;
    int     experts_per_token;    /* how many experts the router selects */

    /* Tiny LRU cache of hot experts in RAM */
    ExpertCache *cache;

    /* Architecture metadata (filled by loader) */
    int     n_layers;
    int     hidden_dim;
    int     vocab_size;
    char    name[128];
};

/* ------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------ */

/* Load a Rabbit-format model from path. Returns NULL on failure. */
RabbitModel *rabbit_load(const char *path);

/* Free everything. */
void rabbit_free(RabbitModel *model);

/* Simple chat loop (placeholder). */
rabbit_status rabbit_chat(RabbitModel *model);

/* Version string. */
const char *rabbit_version(void);

#endif /* RABBIT_H */
