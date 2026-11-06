#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Rabbit %s — local MoE inference for constrained hardware\n"
        "\n"
        "Usage:\n"
        "  %s version\n"
        "  %s chat     --model <path> [options]\n"
        "  %s probe    --model <path> [--prompt text]\n"
        "  %s probe-v2 --model <path> [--prompt text] [--tokens n] [--tokenizer path]\n"
        "  %s stats    --model <path>\n"
        "  %s bench    --model <path> [options]\n"
        "\n"
        "Options:\n"
        "  --model <path>       Path to .rabbit model\n"
        "  --tokenizer <path>   BPE tokenizer (.tok / RBTTOK01)\n"
        "  --vocab <path>       Simple word vocabulary file\n"
        "  --tokens <n>         Maximum new tokens (default: 64 for chat, 32 for probe/bench)\n"
        "  --temp <f>           Sampling temperature (default: 0.85)\n"
        "  --prompt <text>      Prompt for probe / bench (default: hi)\n"
        "  --runs <n>           Number of timed runs for bench (default: 5)\n"
        "  -h, --help           Show this help\n"
        "\n"
        "Examples:\n"
        "  make && make test-model\n"
        "  ./rabbit chat --model models/tiny-test.rabbit --tokens 32\n"
        "  ./rabbit stats --model models/tiny-test.rabbit\n"
        "  ./rabbit bench --model models/ref-v2.rabbit --tokens 16 --runs 10\n"
        "  ./rabbit probe-v2 --model models/mixtral-micro.rabbit --prompt hi --tokens 8\n",
        rabbit_version(), prog, prog, prog, prog, prog, prog);
}

static void try_load_tokenizer(const char *tok_path, const char *vocab_path) {
    if (tok_path) {
        if (rabbit_tokenizer_load_bpe(tok_path) == 0) {
            fprintf(stderr, "tokenizer: BPE loaded (%d tokens)\n", rabbit_tokenizer_vocab());
            return;
        }
        fprintf(stderr, "warning: could not load BPE %s\n", tok_path);
    }
    if (vocab_path) {
        if (rabbit_tokenizer_load_vocab(vocab_path) == 0) {
            fprintf(stderr, "tokenizer: word vocab loaded (%d)\n", rabbit_tokenizer_vocab());
            return;
        }
        fprintf(stderr, "warning: could not load vocab %s (byte mode)\n", vocab_path);
    }
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int cmd_probe(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = "hi";

    for (int i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "--model") == 0) && i + 1 < argc)
            model_path = argv[++i];
        else if ((strcmp(argv[i], "--prompt") == 0) && i + 1 < argc)
            prompt = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }
    if (!model_path) {
        fprintf(stderr, "error: --model required\n");
        return 1;
    }

    RabbitModel *m = rabbit_load(model_path);
    if (!m) return 1;

    int dim = m->hidden_dim;
    float *h = (float *)rabbit_xcalloc((size_t)dim, sizeof(float));

    if (rabbit_ref_forward(m, prompt, h) != 0) {
        fprintf(stderr, "error: ref_forward failed\n");
        free(h);
        rabbit_free(m);
        return 1;
    }

    double sum = 0.0, l1 = 0.0, l2 = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += h[i];
        l1 += fabs((double)h[i]);
        l2 += (double)h[i] * (double)h[i];
    }
    l2 = sqrt(l2);

    printf("PROBE prompt=%s dim=%d sum=%.8f l1=%.8f l2=%.8f first8=", prompt, dim, sum, l1, l2);
    for (int i = 0; i < dim && i < 8; i++) {
        if (i) printf(",");
        printf("%.8f", h[i]);
    }
    printf("\n");

    free(h);
    rabbit_free(m);
    return 0;
}

static int cmd_probe_v2(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = "hi";
    const char *tok_path = NULL;
    int n_new = 32;

    for (int i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "--model") == 0) && i + 1 < argc)
            model_path = argv[++i];
        else if ((strcmp(argv[i], "--prompt") == 0) && i + 1 < argc)
            prompt = argv[++i];
        else if ((strcmp(argv[i], "--tokenizer") == 0) && i + 1 < argc)
            tok_path = argv[++i];
        else if ((strcmp(argv[i], "--tokens") == 0) && i + 1 < argc) {
            n_new = atoi(argv[++i]);
            if (n_new < 1) n_new = 1;
            if (n_new > 128) n_new = 128;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }
    if (!model_path) {
        fprintf(stderr, "error: --model required\n");
        return 1;
    }

    try_load_tokenizer(tok_path, NULL);

    RabbitModel *m = rabbit_load(model_path);
    if (!m) return 1;

    int max_ids = 512 + n_new;
    int *ids = (int *)rabbit_xmalloc(sizeof(int) * (size_t)max_ids);

    int n = rabbit_ref_v2_greedy(m, prompt, n_new, ids, max_ids);
    if (n < 0) {
        fprintf(stderr, "error: ref-v2 greedy failed (is this a ref-v2 model?)\n");
        free(ids);
        rabbit_free(m);
        return 1;
    }

    printf("PROBE_V2 prompt=%s n=%d ids=", prompt, n);
    for (int i = 0; i < n; i++) {
        if (i) printf(",");
        printf("%d", ids[i]);
    }
    printf("\n");

    free(ids);
    rabbit_free(m);
    return 0;
}

static int cmd_bench(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = "hi";
    const char *tok_path = NULL;
    int n_new = 32;
    int runs = 5;

    for (int i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "--model") == 0) && i + 1 < argc)
            model_path = argv[++i];
        else if ((strcmp(argv[i], "--prompt") == 0) && i + 1 < argc)
            prompt = argv[++i];
        else if ((strcmp(argv[i], "--tokenizer") == 0) && i + 1 < argc)
            tok_path = argv[++i];
        else if ((strcmp(argv[i], "--tokens") == 0) && i + 1 < argc) {
            n_new = atoi(argv[++i]);
            if (n_new < 1) n_new = 1;
            if (n_new > 256) n_new = 256;
        } else if ((strcmp(argv[i], "--runs") == 0) && i + 1 < argc) {
            runs = atoi(argv[++i]);
            if (runs < 1) runs = 1;
            if (runs > 100) runs = 100;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }
    if (!model_path) {
        fprintf(stderr, "error: --model required\n");
        return 1;
    }

    try_load_tokenizer(tok_path, NULL);

    RabbitModel *m = rabbit_load(model_path);
    if (!m) return 1;

    int max_ids = 512 + n_new;
    int *ids = (int *)rabbit_xmalloc(sizeof(int) * (size_t)max_ids);

    /* Warm-up */
    int n = rabbit_ref_v2_greedy(m, prompt, n_new, ids, max_ids);
    int use_v2 = (n >= 0);

    if (!use_v2) {
        float *h = (float *)rabbit_xcalloc((size_t)m->hidden_dim, sizeof(float));
        double total_ms = 0.0;
        for (int r = 0; r < runs; r++) {
            double t0 = now_ms();
            if (rabbit_ref_forward(m, prompt, h) != 0) {
                fprintf(stderr, "error: ref_forward failed on run %d\n", r + 1);
                free(h);
                free(ids);
                rabbit_free(m);
                return 1;
            }
            total_ms += now_ms() - t0;
        }
        double avg = total_ms / (double)runs;
        printf("BENCH mode=probe runs=%d avg_ms=%.3f\n", runs, avg);
        free(h);
        free(ids);
        rabbit_free(m);
        return 0;
    }

    double total_ms = 0.0;
    int total_tokens = 0;

    for (int r = 0; r < runs; r++) {
        double t0 = now_ms();
        n = rabbit_ref_v2_greedy(m, prompt, n_new, ids, max_ids);
        double dt = now_ms() - t0;
        if (n < 0) {
            fprintf(stderr, "error: ref-v2 greedy failed on run %d\n", r + 1);
            free(ids);
            rabbit_free(m);
            return 1;
        }
        total_ms += dt;
        total_tokens += n;
    }

    double avg_ms = total_ms / (double)runs;
    double toks_per_sec = (total_ms > 0.0) ? ((double)total_tokens * 1000.0 / total_ms) : 0.0;

    printf("BENCH mode=ref-v2 runs=%d tokens=%d avg_ms=%.3f tokens/sec=%.2f\n",
           runs, total_tokens / runs, avg_ms, toks_per_sec);

    free(ids);
    rabbit_free(m);
    return 0;
}

static int cmd_stats(int argc, char **argv) {
    const char *model_path = NULL;

    for (int i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "--model") == 0) && i + 1 < argc)
            model_path = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }
    if (!model_path) {
        fprintf(stderr, "error: --model required\n");
        return 1;
    }

    RabbitModel *m = rabbit_load(model_path);
    if (!m) return 1;

    RabbitMemoryReport r;
    rabbit_memory_report(m, &r);

    uint32_t n_heads = 0, n_kv = 0, head_dim = 0, n_shared = 0;
    memcpy(&n_heads, m->hdr.reserved + 0, 4);
    memcpy(&n_kv, m->hdr.reserved + 4, 4);
    memcpy(&head_dim, m->hdr.reserved + 8, 4);
    memcpy(&n_shared, m->hdr.reserved + 12, 4);
    if (n_shared > 64) n_shared = 0;

    char name_esc[160];
    size_t j = 0;
    for (size_t i = 0; m->name[i] && j + 2 < sizeof(name_esc); i++) {
        char c = m->name[i];
        if (c == '"' || c == '\\') {
            name_esc[j++] = '\\';
            name_esc[j++] = c;
        } else if ((unsigned char)c < 0x20) {
            continue;
        } else {
            name_esc[j++] = c;
        }
    }
    name_esc[j] = 0;

    /* quant 0 = f32 — do not rewrite to 16 */
    unsigned dense_q = m->hdr.dense_quant;
    unsigned expert_q = m->hdr.expert_quant;

    printf("{\n");
    printf("  \"version\": \"%s\",\n", rabbit_version());
    printf("  \"model_name\": \"%s\",\n", name_esc);
    printf("  \"layers\": %d,\n", m->n_layers);
    printf("  \"hidden_dim\": %d,\n", m->hidden_dim);
    printf("  \"vocab_size\": %d,\n", m->vocab_size);
    printf("  \"n_experts\": %d,\n", m->num_experts);
    printf("  \"n_shared_experts\": %u,\n", n_shared);
    printf("  \"experts_per_token\": %d,\n", m->experts_per_token);
    printf("  \"experts_per_layer\": %d,\n", m->experts_per_layer);
    printf("  \"n_expert_records\": %d,\n", m->n_expert_records);
    printf("  \"experts_shared_table\": %s,\n", m->experts_shared ? "true" : "false");
    printf("  \"expert_dim\": %u,\n", m->hdr.expert_dim);
    printf("  \"n_heads\": %u,\n", n_heads);
    printf("  \"n_kv_heads\": %u,\n", n_kv);
    printf("  \"head_dim\": %u,\n", head_dim);
    printf("  \"dense_bytes\": %zu,\n", r.dense_bytes);
    printf("  \"expert_size\": %zu,\n", m->expert_size);
    printf("  \"total_experts_bytes\": %llu,\n",
           (unsigned long long)m->hdr.total_experts_size);
    printf("  \"dense_quant\": %u,\n", dense_q);
    printf("  \"expert_quant\": %u,\n", expert_q);
    printf("  \"expert_mapped\": %s,\n", m->expert_map ? "true" : "false");
    printf("  \"cache_slots\": %d,\n", r.cache_slots);
    printf("  \"cache_bytes\": %zu,\n", r.cache_bytes);
    printf("  \"kv_bytes_est\": %zu,\n", r.kv_bytes);
    printf("  \"mem_budget_bytes\": %zu,\n", r.budget_bytes);
    printf("  \"mem_total_est_bytes\": %zu,\n", r.total_estimate);
    printf("  \"over_budget\": %s\n", r.over_budget ? "true" : "false");
    printf("}\n");

    free(m->dense);
    m->dense = NULL;
    rabbit_munmap_file(m->expert_map_base, m->expert_map_size, m->expert_fd);
    m->expert_map_base = NULL;
    rabbit_cache_free(m->cache);
    m->cache = NULL;
    rabbit_attn_reset();
    free(m);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "version") == 0) {
        printf("Rabbit %s\n", rabbit_version());
        return 0;
    }

    if (strcmp(argv[1], "help") == 0 ||
        strcmp(argv[1], "-h") == 0 ||
        strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
        return 0;
    }

    if (strcmp(argv[1], "probe") == 0)
        return cmd_probe(argc, argv);

    if (strcmp(argv[1], "probe-v2") == 0)
        return cmd_probe_v2(argc, argv);

    if (strcmp(argv[1], "bench") == 0)
        return cmd_bench(argc, argv);

    if (strcmp(argv[1], "stats") == 0)
        return cmd_stats(argc, argv);

    if (strcmp(argv[1], "chat") == 0) {
        const char *model_path = NULL;
        const char *vocab_path = NULL;
        const char *tok_path = NULL;
        int max_tokens = 64;
        float temperature = 0.85f;

        for (int i = 2; i < argc; i++) {
            if ((strcmp(argv[i], "--model") == 0) && i + 1 < argc)
                model_path = argv[++i];
            else if ((strcmp(argv[i], "--vocab") == 0) && i + 1 < argc)
                vocab_path = argv[++i];
            else if ((strcmp(argv[i], "--tokenizer") == 0) && i + 1 < argc)
                tok_path = argv[++i];
            else if ((strcmp(argv[i], "--tokens") == 0) && i + 1 < argc) {
                max_tokens = atoi(argv[++i]);
                if (max_tokens < 1) max_tokens = 1;
                if (max_tokens > 2048) max_tokens = 2048;
            } else if ((strcmp(argv[i], "--temp") == 0) && i + 1 < argc) {
                temperature = (float)atof(argv[++i]);
                if (temperature < 0.0f) temperature = 0.0f;
                if (temperature > 2.0f) temperature = 2.0f;
            } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                print_usage(argv[0]);
                return 0;
            } else {
                fprintf(stderr, "Unknown option: %s\n\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        }

        if (!model_path) {
            fprintf(stderr, "error: --model <path> is required\n\n");
            print_usage(argv[0]);
            return 1;
        }

        fprintf(stderr, "Rabbit %s\n", rabbit_version());
        fprintf(stderr, "loading %s\n", model_path);

        try_load_tokenizer(tok_path, vocab_path);

        RabbitModel *model = rabbit_load(model_path);
        if (!model) {
            fprintf(stderr, "error: failed to load model\n");
            return 1;
        }

        rabbit_status st = rabbit_chat_ex(model, max_tokens, temperature);
        rabbit_free(model);
        return st == RABBIT_OK ? 0 : 1;
    }

    fprintf(stderr, "Unknown command: %s\n\n", argv[1]);
    print_usage(argv[0]);
    return 1;
}
