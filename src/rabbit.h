/* rabbit.h — public API for the Rabbit MoE engine */
#ifndef RABBIT_H
#define RABBIT_H

#include <stddef.h>
#include <stdint.h>

#define RABBIT_VERSION "0.4.16"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations / opaque types if needed */
typedef struct RabbitModel RabbitModel;
typedef struct ExpertCache ExpertCache;

/* Model load / free */
RabbitModel *rabbit_load(const char *path);
void         rabbit_free(RabbitModel *m);

/* Info / stats */
int  rabbit_n_layers(const RabbitModel *m);
int  rabbit_hidden_dim(const RabbitModel *m);
int  rabbit_n_experts(const RabbitModel *m);
int  rabbit_n_shared_experts(const RabbitModel *m);
int  rabbit_top_k(const RabbitModel *m);
int  rabbit_vocab_size(const RabbitModel *m);
void rabbit_print_stats(const RabbitModel *m);

/* Chat / generation */
int rabbit_chat(RabbitModel *m, const char *prompt, int max_tokens, float temperature);
int rabbit_chat_ex(RabbitModel *m, const char *prompt, int max_tokens, float temperature, float top_p);
int rabbit_chat_ex2(RabbitModel *m, const char *prompt, int max_tokens, float temperature, float top_p, uint64_t seed);

/* Probe / debug */
int rabbit_probe_v2(RabbitModel *m, const char *prompt, int tokens);

/* Cache */
ExpertCache *rabbit_cache_create(size_t budget_bytes, int n_slots);
void         rabbit_cache_free(ExpertCache *c);
int          rabbit_cache_get(ExpertCache *c, int expert_id, void **out_ptr);
int          rabbit_cache_put(ExpertCache *c, int expert_id, const void *src);
void         rabbit_cache_stats(const ExpertCache *c, uint64_t *hits, uint64_t *misses, uint64_t *inserts);

/* Mapping */
int          rabbit_mmap_file(const char *path, void **base, size_t *size, int *fd);
void         rabbit_munmap_file(void *base, size_t size, int fd);
const void  *rabbit_expert_ptr(const RabbitModel *m, int global_expert_id);
const void  *rabbit_expert_ptr_layer(const RabbitModel *m, int layer, int local_expert_id);
int          rabbit_expert_global_id(const RabbitModel *m, int layer, int local_expert_id);

/* MoE / layers */
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

#ifdef __cplusplus
}
#endif

#endif /* RABBIT_H */
