#include "rabbit.h"
#include "validate.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>

/* Validation result codes */
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

/* Configuration for validation run */
typedef struct {
    const char *rabbit_model_path;
    const char *hf_model_dir;
    const char *prompt;
    int max_new_tokens;
    float temperature;
    float rtol;
    float atol;
    int verbose;
    const char *python_exe;      /* Python interpreter */
    const char *hf_backend_script; /* Path to hf_backend.py */
} RabbitValidateConfig;

/* Result of a validation run */
typedef struct {
    int tokens_generated;
    int tokens_matched;
    int tokens_mismatched;
    float max_logit_diff;
    float mean_logit_diff;
    const char *first_mismatch_token;
    int first_mismatch_pos;
} RabbitValidateResult;

/* Compare two logit vectors with relative/absolute tolerance */
static int logits_close(const float *a, const float *b, int n, float rtol, float atol,
                        float *max_diff_out, float *mean_diff_out) {
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int count = 0;

    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        float abs_b = fabsf(b[i]);
        float tol = atol + rtol * abs_b;

        if (diff > tol) {
            return 0;
        }
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
        count++;
    }

    if (max_diff_out) *max_diff_out = max_diff;
    if (mean_diff_out && count > 0) *mean_diff_out = sum_diff / count;
    return 1;
}

/* Call Python HF backend to generate tokens and logits */
static int call_hf_backend(const RabbitValidateConfig *cfg,
                           int **out_ids, float **out_logits, int *out_n,
                           int *out_vocab_size) {
    char cmd[4096];
    char output_file[512];
    snprintf(output_file, sizeof(output_file), "/tmp/rabbit_hf_output_%d.json", getpid());

    const char *py = cfg->python_exe ? cfg->python_exe : "python3";
    const char *script = cfg->hf_backend_script ? cfg->hf_backend_script : "tools/hf_backend.py";

    int written = snprintf(cmd, sizeof(cmd),
        "%s %s --model-dir %s --prompt \"%s\" --max-new-tokens %d --temperature %f --output %s",
        py, script, cfg->hf_model_dir, cfg->prompt, cfg->max_new_tokens, cfg->temperature, output_file);

    if (written >= (int)sizeof(cmd) - 1) {
        fprintf(stderr, "validate: command too long\n");
        return -1;
    }

    if (cfg->verbose) {
        fprintf(stderr, "validate: running HF backend: %s\n", cmd);
    }

    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "validate: HF backend failed with code %d\n", rc);
        return -1;
    }

    /* Parse JSON output */
    FILE *f = fopen(output_file, "r");
    if (!f) {
        fprintf(stderr, "validate: failed to read HF output: %s\n", strerror(errno));
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = (char *)malloc(fsize + 1);
    if (!json) {
        fclose(f);
        return -1;
    }
    fread(json, 1, fsize, f);
    json[fsize] = '\0';
    fclose(f);

    /* Simple JSON parsing for token_ids, logits, vocab_size */
    int *ids = NULL;
    float *logits = NULL;
    int n_ids = 0;
    int vocab_size = 0;

    /* Find token_ids array */
    char *token_ids_start = strstr(json, "\"token_ids\"");
    if (token_ids_start) {
        token_ids_start = strchr(token_ids_start, '[');
        if (token_ids_start) {
            char *p = token_ids_start + 1;
            int capacity = 32;
            ids = (int *)malloc(capacity * sizeof(int));
            while (*p && *p != ']') {
                if (*p >= '0' && *p <= '9' || *p == '-') {
                    if (n_ids >= capacity) {
                        capacity *= 2;
                        ids = (int *)realloc(ids, capacity * sizeof(int));
                    }
                    ids[n_ids++] = atoi(p);
                    while (*p && (*p >= '0' && *p <= '9' || *p == '-')) p++;
                }
                if (*p == ',') p++;
            }
        }
    }

    /* Find logits array */
    char *logits_start = strstr(json, "\"logits\"");
    if (logits_start) {
        logits_start = strchr(logits_start, '[');
        if (logits_start) {
            char *p = logits_start + 1;
            int capacity = 1024;
            logits = (float *)malloc(capacity * sizeof(float));
            int n_logits = 0;
            while (*p && *p != ']') {
                if ((*p >= '0' && *p <= '9') || *p == '-' || *p == '.') {
                    if (n_logits >= capacity) {
                        capacity *= 2;
                        logits = (float *)realloc(logits, capacity * sizeof(float));
                    }
                    logits[n_logits++] = (float)atof(p);
                    while (*p && ((*p >= '0' && *p <= '9') || *p == '-' || *p == '.')) p++;
                }
                if (*p == ',') p++;
            }
            /* logits should be n_ids * vocab_size */
        }
    }

    /* Find vocab_size */
    char *vocab_start = strstr(json, "\"vocab_size\"");
    if (vocab_start) {
        vocab_start = strchr(vocab_start, ':');
        if (vocab_start) {
            vocab_size = atoi(vocab_start + 1);
        }
    }

    free(json);
    unlink(output_file);

    if (!ids || n_ids == 0) {
        free(ids);
        free(logits);
        return -1;
    }

    *out_ids = ids;
    *out_logits = logits;
    *out_n = n_ids;
    *out_vocab_size = vocab_size;
    return 0;
}

/* Run token-exact validation: compare Rabbit greedy decode vs HF greedy decode */
rabbit_validate_status rabbit_validate_token_exact(const RabbitValidateConfig *cfg,
                                                    RabbitValidateResult *result) {
    if (!cfg || !cfg->rabbit_model_path || !cfg->hf_model_dir || !cfg->prompt) {
        return RABBIT_VALIDATE_ERR_ARGS;
    }

    /* Load Rabbit model */
    RabbitModel *rabbit = rabbit_load(cfg->rabbit_model_path);
    if (!rabbit) {
        fprintf(stderr, "validate: failed to load Rabbit model: %s\n", cfg->rabbit_model_path);
        return RABBIT_VALIDATE_ERR_MODEL_LOAD;
    }

    /* Generate with Rabbit (greedy) */
    int max_ids = 512 + cfg->max_new_tokens;
    int *rabbit_ids = (int *)malloc(sizeof(int) * (size_t)max_ids);
    float *rabbit_logits = (float *)malloc(sizeof(float) * (size_t)max_ids * (size_t)rabbit->vocab_size);

    if (!rabbit_ids || !rabbit_logits) {
        fprintf(stderr, "validate: OOM allocating buffers\n");
        free(rabbit_ids);
        free(rabbit_logits);
        rabbit_free(rabbit);
        return RABBIT_VALIDATE_ERR_MEM;
    }

    int rabbit_n = rabbit_ref_v2_greedy(rabbit, cfg->prompt, cfg->max_new_tokens,
                                         rabbit_ids, max_ids);
    if (rabbit_n < 0) {
        fprintf(stderr, "validate: rabbit_ref_v2_greedy failed\n");
        free(rabbit_ids);
        free(rabbit_logits);
        rabbit_free(rabbit);
        return RABBIT_VALIDATE_ERR_IO;
    }

    /* Generate with HF backend */
    int *hf_ids = NULL;
    float *hf_logits = NULL;
    int hf_n = 0;
    int hf_vocab_size = 0;

    if (call_hf_backend(cfg, &hf_ids, &hf_logits, &hf_n, &hf_vocab_size) != 0) {
        fprintf(stderr, "validate: HF backend generation failed\n");
        free(rabbit_ids);
        free(rabbit_logits);
        rabbit_free(rabbit);
        return RABBIT_VALIDATE_ERR_HF_BACKEND;
    }

    /* Compare token-by-token */
    int matched = 0;
    int mismatched = 0;
    int first_mismatch = -1;
    float max_logit_diff = 0.0f;
    float mean_logit_diff = 0.0f;

    int n_compare = (rabbit_n < hf_n) ? rabbit_n : hf_n;

    for (int i = 0; i < n_compare; i++) {
        if (rabbit_ids[i] == hf_ids[i]) {
            matched++;
        } else {
            mismatched++;
            if (first_mismatch == -1) {
                first_mismatch = i;
            }
        }

        /* Compare logits for this position if available */
        if (rabbit_logits && hf_logits && hf_vocab_size > 0) {
            float *r_logits = &rabbit_logits[(size_t)i * rabbit->vocab_size];
            float *h_logits = &hf_logits[(size_t)i * hf_vocab_size];
            int vocab_compare = (rabbit->vocab_size < hf_vocab_size) ? rabbit->vocab_size : hf_vocab_size;
            float md, mn;
            if (logits_close(r_logits, h_logits, vocab_compare,
                             cfg->rtol, cfg->atol, &md, &mn)) {
                if (md > max_logit_diff) max_logit_diff = md;
                mean_logit_diff += mn;
            } else {
                mismatched++;
                if (first_mismatch == -1) first_mismatch = i;
            }
        }
    }

    if (n_compare > 0 && mean_logit_diff > 0) {
        mean_logit_diff /= n_compare;
    }

    /* Cleanup */
    free(rabbit_ids);
    free(rabbit_logits);
    free(hf_ids);
    free(hf_logits);
    rabbit_free(rabbit);

    /* Populate result */
    if (result) {
        result->tokens_generated = rabbit_n;
        result->tokens_matched = matched;
        result->tokens_mismatched = mismatched;
        result->max_logit_diff = max_logit_diff;
        result->mean_logit_diff = mean_logit_diff;
        result->first_mismatch_pos = first_mismatch;
        result->first_mismatch_token = (first_mismatch >= 0) ? "mismatch" : NULL;
    }

    if (mismatched > 0) {
        if (cfg->verbose) {
            fprintf(stderr, "validate: FAIL - %d/%d tokens mismatched (first at pos %d)\n",
                    mismatched, n_compare, first_mismatch);
        }
        return RABBIT_VALIDATE_ERR_TOKEN_MISMATCH;
    }

    if (cfg->verbose) {
        fprintf(stderr, "validate: OK - %d/%d tokens matched, max_logit_diff=%.6f, mean=%.6f\n",
                matched, n_compare, max_logit_diff, mean_logit_diff);
    }

    return RABBIT_VALIDATE_OK;
}

/* CLI entry point for `rabbit validate` command */
int rabbit_cmd_validate(int argc, char **argv) {
    RabbitValidateConfig cfg = {
        .rabbit_model_path = NULL,
        .hf_model_dir = NULL,
        .prompt = "hi",
        .max_new_tokens = 32,
        .temperature = 0.0f,
        .rtol = 1e-4f,
        .atol = 1e-5f,
        .verbose = 1,
        .python_exe = "python3",
        .hf_backend_script = "tools/hf_backend.py"
    };

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            cfg.rabbit_model_path = argv[++i];
        } else if (strcmp(argv[i], "--hf-model") == 0 && i + 1 < argc) {
            cfg.hf_model_dir = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            cfg.prompt = argv[++i];
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            cfg.max_new_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            cfg.temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--rtol") == 0 && i + 1 < argc) {
            cfg.rtol = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--atol") == 0 && i + 1 < argc) {
            cfg.atol = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--python") == 0 && i + 1 < argc) {
            cfg.python_exe = argv[++i];
        } else if (strcmp(argv[i], "--hf-backend") == 0 && i + 1 < argc) {
            cfg.hf_backend_script = argv[++i];
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            cfg.verbose = 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            fprintf(stderr,
                "Usage: rabbit validate --model <rabbit> --hf-model <hf_dir> [options]\n"
                "Options:\n"
                "  --model <path>       Rabbit .rabbit model (required)\n"
                "  --hf-model <dir>     HF model directory (required)\n"
                "  --prompt <text>      Prompt (default: \"hi\")\n"
                "  --tokens <n>         Max new tokens (default: 32)\n"
                "  --temp <f>           Temperature (default: 0.0 = greedy)\n"
                "  --rtol <f>           Relative tolerance (default: 1e-4)\n"
                "  --atol <f>           Absolute tolerance (default: 1e-5)\n"
                "  --python <path>      Python interpreter (default: python3)\n"
                "  --hf-backend <path>  HF backend script (default: tools/hf_backend.py)\n"
                "  -q, --quiet          Suppress verbose output\n"
            );
            return 0;
        } else {
            fprintf(stderr, "validate: unknown option: %s\n", argv[i]);
            return RABBIT_VALIDATE_ERR_ARGS;
        }
    }

    if (!cfg.rabbit_model_path || !cfg.hf_model_dir) {
        fprintf(stderr, "validate: --model and --hf-model are required\n");
        return RABBIT_VALIDATE_ERR_ARGS;
    }

    RabbitValidateResult result;
    rabbit_validate_status status = rabbit_validate_token_exact(&cfg, &result);

    if (status == RABBIT_VALIDATE_OK) {
        printf("VALIDATE OK: %d tokens matched, max_logit_diff=%.6f, mean_logit_diff=%.6f\n",
               result.tokens_matched, result.max_logit_diff, result.mean_logit_diff);
        return 0;
    } else {
        printf("VALIDATE FAIL: code=%d, matched=%d, mismatched=%d, first_mismatch=%d\n",
               status, result.tokens_matched, result.tokens_mismatched, result.first_mismatch_pos);
        return 1;
    }
}