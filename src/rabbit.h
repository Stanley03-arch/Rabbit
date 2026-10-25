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
#define RABBIT_MAGIC   "RABBIT01"

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
 * On-disk header (must stay packed / little-endian)
 * ------------------------------------------------------------ */
#pragma pack(push, 1)
typedef struct {
    char     magic[8];          /* "RABBIT01" */
    uint32_t version;
    uint32_t header_size;

    uint32_t n_layers;
    uint32_t hidden_dim;
    uint32_t vocab_size;
    uint32_t n_experts;
    uint32_t experts_per_token;
    uint32_t expert_dim;

    uint64_t dense_size;
    uint64_t expert_size;         /* size of one expert */
    uint64_t total_experts_size;

    uint32_t dense_quant;         /* 0=f16, 4=int4, ... */
    uint32_t expert_quant;

    char     name[64];
    uint8_t  reserved[128];
} RabbitHeader;
#pragma pack(pop)

/* ------------------------------------------------------------
 * Model handle (in-memory)
 * ------------------------------------------------------------ */
struct RabbitModel {
    RabbitHeader hdr;

    /* Always-resident dense / shared part */
    void   *dense;
    size_t  dense_size;

    /* Expert storage (memory-mapped file) */
    int     expert_fd;
    void   *expert_map;
    size_t  expert_map_size;
    size_t  expert_size;          /* size of one expert in bytes */
    int     num_experts;
    int     experts_per_token;

    /* Tiny LRU cache of hot experts in RAM */
    ExpertCache *cache;

    /* Architecture metadata (convenience) */
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

/* Print model info. */
void rabbit_print_info(const RabbitModel *model);

#endif /* RABBIT_H */
