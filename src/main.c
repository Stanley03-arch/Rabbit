#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Rabbit %s \ud83d\udc07\n"
        "\n"
        "Usage:\n"
        "  %s chat --model <path> [options]\n"
        "  %s version\n"
        "\n"
        "Options:\n"
        "  --model <path>    Path to .rabbit model (required)\n"
        "  --vocab <path>    Optional word vocab file (one token per line)\n"
        "  --tokens <n>      Max new tokens to generate (default 64)\n"
        "  --temp <f>        Sampling temperature (default 0.85)\n"
        "\n"
        "Examples:\n"
        "  %s chat --model models/demo-moe.rabbit\n"
        "  %s chat --model models/demo-moe.rabbit --tokens 128 --temp 0.7\n"
        "\n"
        "Tiny engine for large MoE models on constrained hardware.\n",
        rabbit_version(), prog, prog, prog, prog);
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

    if (strcmp(argv[1], "chat") == 0) {
        const char *model_path = NULL;
        const char *vocab_path = NULL;
        int max_tokens = 64;
        float temperature = 0.85f;

        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_path = argv[++i];
            } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
                vocab_path = argv[++i];
            } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
                max_tokens = atoi(argv[++i]);
                if (max_tokens < 1) max_tokens = 1;
                if (max_tokens > 2048) max_tokens = 2048;
            } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
                temperature = (float)atof(argv[++i]);
                if (temperature < 0.0f) temperature = 0.0f;
                if (temperature > 2.0f) temperature = 2.0f;
            } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                print_usage(argv[0]);
                return 0;
            }
        }

        if (!model_path) {
            fprintf(stderr, "Error: --model <path> is required\n\n");
            print_usage(argv[0]);
            return 1;
        }

        printf("Rabbit %s \ud83d\udc07\n", rabbit_version());
        printf("Loading model from: %s\n", model_path);

        if (vocab_path) {
            if (rabbit_tokenizer_load_vocab(vocab_path) != 0) {
                fprintf(stderr, "[warn] could not load vocab %s — using byte mode\n", vocab_path);
                rabbit_tokenizer_set_byte_mode();
            }
        } else {
            rabbit_tokenizer_set_byte_mode();
        }

        RabbitModel *model = rabbit_load(model_path);
        if (!model) {
            fprintf(stderr, "Failed to load model.\n");
            return 1;
        }

        rabbit_status st = rabbit_chat_ex(model, max_tokens, temperature);
        rabbit_free(model);
        return st == RABBIT_OK ? 0 : 1;
    }

    print_usage(argv[0]);
    return 1;
}
