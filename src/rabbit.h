#ifndef RABBIT_H
#define RABBIT_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <sys/types.h>

#define RABBIT_VERSION "0.4.16"
#define RABBIT_MAGIC   "RABBIT01"
#define RABBIT_MAX_CACHE_SLOTS 8
#define RABBIT_INT4_GROUP 32

#define RABBIT_DEFAULT_MEM_BUDGET_MB 1536

typedef enum {
    RABBIT_OK = 0,
    RABBIT_ERR_IO,
    RABBIT_ERR_MEM,
    RABBIT_ERR_FORMAT,
    RABBIT_ERR_ARG,
    RABBIT_ERR_UNSUPPORTED
} RabbitStatus;

typedef struct RabbitHeader {
    char     magic[8];
    uint32_t version;
    uint32_t hidden_dim;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t intermediate_size;
    uint32_t n_experts;
    uint32_t top_k;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    uint32_t expert_quant;      /* 0=f32, 4=group-int4, 8=int8 */
    uint32_t n_shared_experts;  /* DeepSeek-style; 0 = none */
    float    rope_theta;
    float    rms_eps;
    uint32_t reserved[16];
} RabbitHeader;

typedef struct ExpertCache ExpertCache;
typedef struct RabbitModel RabbitModel;

/* Load / free */
RabbitModel *rabbit_load(const char *path);
void         rabbit_free(RabbitModel *m);
RabbitStatus rabbit_status(const RabbitModel *m);

/* Metadata accessors */
const RabbitHeader *rabbit_header(const RabbitModel *m);
int  rabbit_n_layers(const RabbitModel *m);
int  rabbit_hidden_dim(const RabbitModel *m);
int  rabbit_n_experts(const RabbitModel *m);
int  rabbit_n_shared_experts(const RabbitModel *m);
int  rabbit_top_k(const RabbitModel *m);
int  rabbit_vocab_size(const RabbitModel *m);
void rabbit_print_stats(const RabbitModel *m);

/* Generation */
int rabbit_chat(RabbitModel *m, const char *prompt, int max_tokens, float temperature);
int rabbit_chat_ex(RabbitModel *m, const char *prompt, int max_tokens, float temperature, float top_p);
int rabbit_chat_ex2(RabbitModel *m, const char *prompt, int max_tokens, float temperature, float top_p, uint64_t seed);

/* Probe / debug path for layout v2 */
int rabbit_probe_v2(RabbitModel *m, const char *prompt, int tokens);

/* Expert cache */
ExpertCache *rabbit_cache_create(size_t budget_bytes, int n_slots);
void         rabbit_cache_free(ExpertCache *c);
int          rabbit_cache_get(ExpertCache *c, int expert_id, void **out_ptr);
int          rabbit_cache_put(ExpertCache *c, int expert_id, const void *src);
void         rabbit_cache_stats(const ExpertCache *c, uint64_t *hits, uint64_t *misses, uint64_t *inserts);

/* mmap helpers */
int          rabbit_mmap_file(const char *path, void **base, size_t *size, int *fd);
void         rabbit_munmap_file(void *base, size_t size, int fd);
const void  *rabbit_expert_ptr(const RabbitModel *m, int global_expert_id);
const void  *rabbit_expert_ptr_layer(const RabbitModel *m, int layer, int local_expert_id);
int          rabbit_expert_global_id(const RabbitModel *m, int layer, int local_expert_id);

/* MoE / layer steps */
const void *rabbit_fetch_expert(RabbitModel *m, int global_expert_id, int promote);
const void *rabbit_fetch_expert_layer(RabbitModel *m, int layer, int local_expert_id, int promote);
int         rabbit_moe_step(RabbitModel *m, float *hidden, int hidden_dim);
int         rabbit_ref_forward(RabbitModel *m, const char *prompt, float *out_hidden);
void        rabbit_attn_reset(void);
int         rabbit_layer_step(RabbitModel *m, float *hidden, int hidden_dim);

/* Phase D: faithful ref-v2 forward */
int rabbit_ref_v2_forward(RabbitModel *m, const int *token_ids, int n_tokens,
                          float *out_hidden, float *out_logits);
int rabbit_ref_v2_greedy(RabbitModel *m, const char *prompt, int n_new,
                         int *out_ids, int max_ids);

/* Quantization */
void rabbit_dequant_int8(const int8_t *src, float *dst, int n, float scale);
void rabbit_dequant_int4(const uint8_t *src, float *dst, int n, float scale);
int  rabbit_pack_int4(const float *src, int n, uint8_t *dst, float *out_scale);

/* Group-scaled int4 (expert_quant == 4) */
int  rabbit_pack_int4_grouped(const float *src, int n, int group_size,
                              uint8_t *dst, int dst_cap);
int  rabbit_dequant_int4_grouped(const uint8_t *src, float *dst, int n);
void rabbit_gemv_int4_grouped(const uint8_t *W, const float *x, float *y,
                              int out_dim, int in_dim);
int  rabbit_int4_grouped_nbytes(int n, int group_size);

/* Tokenizer */
int  rabbit_tokenize(const char *text, int *out_ids, int max_ids);
void rabbit_detokenize(int id, char *out, int out_size);
void rabbit_detokenize_seq(const int *ids, int n, char *out, int out_size);
int  rabbit_tokenizer_vocab(void);
void rabbit_tokenizer_set_byte_mode(void);
int  rabbit_tokenizer_load_vocab(const char *path);
int  rabbit_tokenizer_load_bpe(const char *path);  /* RBTTOK01 .tok file */

#endif /* RABBIT_H */
