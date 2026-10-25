#ifndef RABBIT_H
#define RABBIT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <sys/types.h>

#define RABBIT_VERSION "0.2.4"
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
    void   *expert_map_base;
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

/* Lifecycle */
RabbitModel  *rabbit_load(const char *path);
void          rabbit_free(RabbitModel *model);
rabbit_status rabbit_chat(RabbitModel *model);
rabbit_status rabbit_chat_ex(RabbitModel *model, int max_tokens, float temperature);
const char   *rabbit_version(void);
void          rabbit_print_info(const RabbitModel *model);

/* Cache */
ExpertCache *rabbit_cache_create(int n_slots);
void         rabbit_cache_free(ExpertCache *c);
void         rabbit_cache_set_expert_size(ExpertCache *c, size_t size);
void        *rabbit_cache_get(ExpertCache *c, int expert_id);
void        *rabbit_cache_put(ExpertCache *c, int expert_id, const void *src);
void         rabbit_cache_stats(const ExpertCache *c, uint64_t *hits, uint64_t *misses, uint64_t *inserts);

/* Mapping */
int          rabbit_mmap_file(const char *path, void **base, size_t *size, int *fd);
void         rabbit_munmap_file(void *base, size_t size, int fd);
const void  *rabbit_expert_ptr(const RabbitModel *m, int expert_id);

/* MoE / layers */
const void *rabbit_fetch_expert(RabbitModel *m, int expert_id, int promote);
int         rabbit_moe_step(RabbitModel *m, float *hidden, int hidden_dim);
int         rabbit_ref_forward(RabbitModel *m, const char *prompt, float *out_hidden);
void        rabbit_attn_reset(void);
int         rabbit_layer_step(RabbitModel *m, float *hidden, int hidden_dim);

/* Quantization */
void rabbit_dequant_int8(const int8_t *src, float *dst, int n, float scale);
void rabbit_dequant_int4(const uint8_t *src, float *dst, int n, float scale);
int  rabbit_pack_int4(const float *src, int n, uint8_t *dst, float *out_scale);

/* Tokenizer */
int  rabbit_tokenize(const char *text, int *out_ids, int max_ids);
void rabbit_detokenize(int id, char *out, int out_size);
void rabbit_detokenize_seq(const int *ids, int n, char *out, int out_size);
int  rabbit_tokenizer_vocab(void);
void rabbit_tokenizer_set_byte_mode(void);
int  rabbit_tokenizer_load_vocab(const char *path);

#endif /* RABBIT_H */
