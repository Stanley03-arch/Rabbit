#include "rabbit.h"
#include "validate.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Mock HF model for testing - returns predetermined token sequences */
typedef struct {
    int *ids;
    int n_ids;
    float *logits;
    int vocab_size;
} MockHFModel;

static int mock_hf_load(const char *path, void **out) {
    (void)path;
    MockHFModel *m = (MockHFModel *)calloc(1, sizeof(MockHFModel));
    if (!m) return -1;

    /* Deterministic test sequence */
    int seq[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    m->n_ids = 10;
    m->ids = (int *)malloc(sizeof(int) * 10);
    memcpy(m->ids, seq, sizeof(seq));
    m->vocab_size = 32;
    m->logits = (float *)calloc(10 * 32, sizeof(float));
    *out = m;
    return 0;
}

static int mock_hf_generate(void *model, const char *prompt, int max_new,
                            float temp, int **out_ids, float **out_logits, int *out_n) {
    (void)prompt; (void)temp;
    MockHFModel *m = (MockHFModel *)model;
    int n = (max_new < m->n_ids) ? max_new : m->n_ids;

    *out_ids = (int *)malloc(sizeof(int) * n);
    memcpy(*out_ids, m->ids, sizeof(int) * n);

    *out_logits = (float *)malloc(sizeof(float) * n * m->vocab_size);
    /* Put peak at the correct token */
    for (int i = 0; i < n; i++) {
        float *row = &(*out_logits)[(size_t)i * m->vocab_size];
        for (int v = 0; v < m->vocab_size; v++) {
            row[v] = (v == m->ids[i]) ? 10.0f : 0.1f;
        }
    }
    *out_n = n;
    return 0;
}

static void mock_hf_free(void *model) {
    MockHFModel *m = (MockHFModel *)model;
    free(m->ids);
    free(m->logits);
    free(m);
}

/* Replace the stubs in validate.c for testing */
int load_hf_model(const char *hf_dir, void **hf_model_out) {
    return mock_hf_load(hf_dir, hf_model_out);
}

int hf_generate(void *hf_model, const char *prompt, int max_new_tokens,
                float temperature, int **out_ids, float **out_logits, int *out_n) {
    return mock_hf_generate(hf_model, prompt, max_new_tokens, temperature, out_ids, out_logits, out_n);
}

void hf_free(void *hf_model) {
    mock_hf_free(hf_model);
}

/* Test 1: Perfect match */
static int test_perfect_match() {
    printf("test_perfect_match: ");

    /* Create a minimal ref-v2 model for testing */
    system("python3 tools/create_ref_v2.py --out /tmp/test_validate.rabbit --seed 42 >/dev/null 2>&1");

    RabbitValidateConfig cfg = {
        .rabbit_model_path = "/tmp/test_validate.rabbit",
        .hf_model_dir = "/tmp/mock_hf",
        .prompt = "hi",
        .max_new_tokens = 10,
        .temperature = 0.0f,
        .rtol = 1e-4f,
        .atol = 1e-5f,
        .verbose = 0
    };

    RabbitValidateResult result;
    int status = rabbit_validate_token_exact(&cfg, &result);

    if (status == RABBIT_VALIDATE_OK && result.tokens_matched == 10) {
        printf("PASS\n");
        return 0;
    } else {
        printf("FAIL (status=%d, matched=%d)\n", status, result.tokens_matched);
        return 1;
    }
}

/* Test 2: Token mismatch detection */
static int test_token_mismatch() {
    printf("test_token_mismatch: ");

    /* Create a model that will produce different tokens */
    system("python3 tools/create_ref_v2.py --out /tmp/test_validate2.rabbit --seed 99 >/dev/null 2>&1");

    RabbitValidateConfig cfg = {
        .rabbit_model_path = "/tmp/test_validate2.rabbit",
        .hf_model_dir = "/tmp/mock_hf",
        .prompt = "hi",
        .max_new_tokens = 10,
        .temperature = 0.0f,
        .rtol = 1e-4f,
        .atol = 1e-5f,
        .verbose = 0
    };

    RabbitValidateResult result;
    int status = rabbit_validate_token_exact(&cfg, &result);

    if (status == RABBIT_VALIDATE_ERR_TOKEN_MISMATCH && result.tokens_mismatched > 0) {
        printf("PASS (detected %d mismatches)\n", result.tokens_mismatched);
        return 0;
    } else {
        printf("FAIL (status=%d, expected mismatch)\n", status);
        return 1;
    }
}

/* Test 3: Logit tolerance */
static int test_logit_tolerance() {
    printf("test_logit_tolerance: ");

    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {1.0001f, 2.0001f, 3.0001f, 4.0001f};
    float max_diff, mean_diff;

    int close = logits_close(a, b, 4, 1e-3f, 1e-5f, &max_diff, &mean_diff);
    if (close && max_diff <= 0.0002f) {
        printf("PASS (max_diff=%.6f)\n", max_diff);
        return 0;
    } else {
        printf("FAIL (close=%d, max_diff=%.6f)\n", close, max_diff);
        return 1;
    }
}

/* Test 4: Logit mismatch detection */
static int test_logit_mismatch() {
    printf("test_logit_mismatch: ");

    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {1.1f, 2.0f, 3.0f, 4.0f};  // 10% diff on first element
    float max_diff, mean_diff;

    int close = logits_close(a, b, 4, 1e-3f, 1e-5f, &max_diff, &mean_diff);
    if (!close && max_diff > 0.05f) {
        printf("PASS (detected diff=%.6f)\n", max_diff);
        return 0;
    } else {
        printf("FAIL (close=%d, max_diff=%.6f)\n", close, max_diff);
        return 1;
    }
}

int main() {
    int failed = 0;

    failed += test_logit_tolerance();
    failed += test_logit_mismatch();
    failed += test_perfect_match();
    failed += test_token_mismatch();

    if (failed == 0) {
        printf("\nAll validate tests PASSED\n");
        return 0;
    } else {
        printf("\n%d validate tests FAILED\n", failed);
        return 1;
    }
}