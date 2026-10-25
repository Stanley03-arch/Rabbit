#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
        "\n"
        "Options:\n"
        "  --model <path>       Path to .rabbit model\n"
        "  --tokenizer <path>   BPE tokenizer (.tok / RBTTOK01)\n"
        "  --vocab <path>       Simple word vocabulary file\n"
        "  --tokens <n>         Maximum new tokens (default: 64)\n"
        "  --temp <f>           Sampling temperature (default: 0.85)\n"
        "  --prompt <text>      Prompt for probe (default: hi)\n"
        "  -h, --help           Show this help\n",
        rabbit_version(), prog, prog, prog, prog, prog);
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
    float *h = (float *)calloc((size_t)dim, sizeof(float));
    if (!h) {
        rabbit_free(m);
        return 1;
    }

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
    int *ids = (int *)malloc(sizeof(int) * (size_t)max_ids);
    if (!ids) {
        rabbit_free(m);
        return 1;
    }

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

    printf("{\n");
    printf("  \"version\": \"%s\",\n", rabbit_version());
    printf("  \"model_name\": \"%s\",\n", name_esc);
    printf("  \"layers\": %d,\n", m->n_layers);
    printf("  \"hidden_dim\": %d,\n", m->hidden_dim);
    printf("  \"vocab_size\": %d,\n", m->vocab_size);
    printf("  \"n_experts\": %d,\n", m->num_experts);
    printf("  \"experts_per_token\": %d,\n", m->experts_per_token);
    printf("  \"expert_dim\": %u,\n", m->hdr.expert_dim);
    printf("  \"dense_bytes\": %zu,\n", r.dense_bytes);
    printf("  \"expert_size\": %zu,\n", m->expert_size);
    printf("  \"total_experts_bytes\": %llu,\n",
           (unsigned long long)m->hdr.total_experts_size);
    printf("  \"dense_quant\": %u,\n", m->hdr.dense_quant ? m->hdr.dense_quant : 16);
    printf("  \"expert_quant\": %u,\n", m->hdr.expert_quant ? m->hdr.expert_quant : 16);
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
