#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Rabbit %s \ud83d\udc07\n"
        "Usage:\n"
        "  %s chat --model <path>\n"
        "  %s version\n"
        "\n"
        "Tiny engine for large MoE models on constrained hardware.\n",
        rabbit_version(), prog, prog);
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
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_path = argv[++i];
            }
        }

        if (!model_path) {
            fprintf(stderr, "Error: --model <path> is required\n");
            return 1;
        }

        printf("Rabbit %s \ud83d\udc07\n", rabbit_version());
        printf("Loading model from: %s\n", model_path);

        RabbitModel *model = rabbit_load(model_path);
        if (!model) {
            fprintf(stderr, "Failed to load model.\n");
            return 1;
        }

        rabbit_status st = rabbit_chat(model);
        rabbit_free(model);

        return st == RABBIT_OK ? 0 : 1;
    }

    print_usage(argv[0]);
    return 1;
}
