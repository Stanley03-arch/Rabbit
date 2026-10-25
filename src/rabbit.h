#ifndef RABBIT_H
#define RABBIT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <sys/types.h>

#define RABBIT_VERSION "0.1.4-dev"
#define RABBIT_MAGIC   "RABBIT01"
#define RABBIT_MAX_CACHE_SLOTS 8

typedef enum {
    RABBIT_OK = 0,
    RABBIT_ERR_IO,
    RABBIT_ERR_MEM,
    RABBIT_ERR_FORMAT,
    RABBIT_ERR_UNSUPPORTED,
    RABBIT_ERR_ARGS
} rabbit_status;

typedef struct RabbitModel RabbitModel;
typedef struct ExpertCache ExpertCache;

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

struct RabbitModel {
    RabbitHeader hdr;

    void   *dense;
    size_t  dense_size;

    int     expert_fd;
    void   *expert_map;
    size_t  expert_map_size;
    size_t  expert_size;
    int     num_experts;
    int     experts_per_token;

    ExpertCache *cache;

    int     n_layers;
    int     hidden_dim;
    int     vocab_size;
    char    name[128];
};

/* Public API */
RabbitModel *rabbit_load(const char *path);
void         rabbit_free(RabbitModel *model);
rabbit_status rabbit_chat(RabbitModel *model);
const char  *rabbit_version(void);
void         rabbit_print_info(const RabbitModel *model);

/* Memory / cache */
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

/* MoE */
const void  *rabbit_fetch_expert(RabbitModel *m, int expert_id, int promote);
int          rabbit_moe_step(RabbitModel *m, float *hidden, int hidden_dim);

/* Attention + full layer */
void         rabbit_attn_reset(void);
int          rabbit_layer_step(RabbitModel *m, float *hidden, int hidden_dim);

/* Quant helpers */
void rabbit_dequant_int8(const int8_t *src, float *dst, int n, float scale);
void rabbit_dequant_int4(const uint8_t *src, float *dst, int n, float scale);

/* Tokenizer */
int  rabbit_tokenize(const char *text, int *out_ids, int max_ids);
void rabbit_detokenize(int id, char *out, int out_size);
void rabbit_detokenize_seq(const int *ids, int n, char *out, int out_size);
int  rabbit_tokenizer_vocab(void);

#endif /* RABBIT_H */
