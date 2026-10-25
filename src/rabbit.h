#ifndef RABBIT_H
#define RABBIT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <sys/types.h>

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

RabbitModel *rabbit_load(const char *path);
void         rabbit_free(RabbitModel *model);
rabbit_status rabbit_chat(RabbitModel *model);
const char  *rabbit_version(void);
void         rabbit_print_info(const RabbitModel *model);

/* Memory / cache helpers (used internally and by moe.c) */
ExpertCache *rabbit_cache_create(int n_slots);
void         rabbit_cache_free(ExpertCache *c);
void         rabbit_cache_set_expert_size(ExpertCache *c, size_t size);
void        *rabbit_cache_get(ExpertCache *c, int expert_id);
void        *rabbit_cache_put(ExpertCache *c, int expert_id, const void *src);

int          rabbit_mmap_experts(const char *path, void **map, size_t *size, int *fd);
int          rabbit_mmap_experts_region(const char *path, off_t offset, size_t length,
                                        void **map, int *fd);
void         rabbit_munmap_experts(void *map, size_t size, int fd);
const void  *rabbit_expert_ptr(const RabbitModel *m, int expert_id);

#endif /* RABBIT_H */
