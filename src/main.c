#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Rabbit %s \u2014 local MoE inference for constrained hardware\n"
        "\n"
        "Usage:\n"
        "  %s version\n"
        "  %s chat --model <path> [options]\n"
        "  %s probe --model <path> [--prompt text]\n"
        "\n"
        "Options:\n"
        "  --model <path>   Path to .rabbit model\n"
        "  --vocab <path>   Word vocabulary file\n"
        "  --tokens <n>     Maximum new tokens (default: 64)\n"
        "  --temp <f>       Sampling temperature (default: 0.85)\n"
        "  --prompt <text>  Prompt for probe (default: hi)\n"
        "  -h, --help       Show this help\n",
        rabbit_version(), prog, prog, prog);
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

    /* Machine-readable line for tests */
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

    if (strcmp(argv[1], "chat") == 0) {
        const char *model_path = NULL;
        const char *vocab_path = NULL;
        int max_tokens = 64;
        float temperature = 0.85f;

        for (int i = 2; i < argc; i++) {
            if ((strcmp(argv[i], "--model") == 0) && i + 1 < argc)
                model_path = argv[++i];
            else if ((strcmp(argv[i], "--vocab") == 0) && i + 1 < argc)
                vocab_path = argv[++i];
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

        RabbitModel *model = rabbit_load(model_path);
        if (!model) {
            fprintf(stderr, "error: failed to load model\n");
            return 1;
        }

        if (vocab_path) {
            if (rabbit_tokenizer_load_vocab(vocab_path) != 0)
                fprintf(stderr, "warning: could not load vocab %s (using byte mode)\n", vocab_path);
        }

        rabbit_status st = rabbit_chat_ex(model, max_tokens, temperature);
        rabbit_free(model);
        return st == RABBIT_OK ? 0 : 1;
    }

    fprintf(stderr, "Unknown command: %s\n\n", argv[1]);
    print_usage(argv[0]);
    return 1;
}
