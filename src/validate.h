#ifndef RABBIT_VALIDATE_H
#define RABBIT_VALIDATE_H

#include "rabbit.h"
#include <stddef.h>

typedef enum {
    RABBIT_VALIDATE_OK = 0,
    RABBIT_VALIDATE_ERR_MODEL_LOAD,
    RABBIT_VALIDATE_ERR_HF_LOAD,
    RABBIT_VALIDATE_ERR_TOKEN_MISMATCH,
    RABBIT_VALIDATE_ERR_LOGIT_MISMATCH,
    RABBIT_VALIDATE_ERR_IO,
    RABBIT_VALIDATE_ERR_ARGS,
    RABBIT_VALIDATE_ERR_MEM,
    RABBIT_VALIDATE_ERR_HF_BACKEND
} rabbit_validate_status;

typedef struct {
    const char *rabbit_model_path;
    const char *hf_model_dir;
    const char *prompt;
    int max_new_tokens;
    float temperature;
    float rtol;
    float atol;
    int verbose;
    const char *python_exe;
    const char *hf_backend_script;
} RabbitValidateConfig;

typedef struct {
    int tokens_generated;
    int tokens_matched;
    int tokens_mismatched;
    float max_logit_diff;
    float mean_logit_diff;
    const char *first_mismatch_token;
    int first_mismatch_pos;
} RabbitValidateResult;

/* Core validation: token-exact + logit tolerance vs HF reference */
rabbit_validate_status rabbit_validate_token_exact(const RabbitValidateConfig *cfg,
                                                    RabbitValidateResult *result);

/* CLI command: rabbit validate --model <rabbit> --hf-model <hf_dir> [options] */
int rabbit_cmd_validate(int argc, char **argv);

#endif /* RABBIT_VALIDATE_H */