#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Forward declarations from other modules */
ExpertCache *rabbit_cache_create(int n_slots);
void rabbit_cache_free(ExpertCache *c);
int  rabbit_mmap_experts(const char *path, void **map, size_t *size, int *fd);
void rabbit_munmap_experts(void *map, size_t size, int fd);

const char *rabbit_version(void) {
    return RABBIT_VERSION;
}

RabbitModel *rabbit_load(const char *path) {
    if (!path) return NULL;

    /*
     * Placeholder loader.
     * Real implementation will:
     *   1. Read a Rabbit header (magic, version, architecture)
     *   2. Load dense weights into RAM
     *   3. mmap the expert blob
     *   4. Initialize a tiny expert cache
     */

    printf("[rabbit] model loader is still a stub.\n");
    printf("[rabbit] path: %s\n", path);
    printf("[rabbit] next step: define Rabbit file format + dense loader.\n");

    RabbitModel *m = calloc(1, sizeof(RabbitModel));
    if (!m) return NULL;

    strncpy(m->name, "stub-model", sizeof(m->name) - 1);
    m->expert_fd = -1;
    m->cache = rabbit_cache_create(4);  /* very small cache for 4GB target */

    return m;
}

void rabbit_free(RabbitModel *model) {
    if (!model) return;

    free(model->dense);
    rabbit_munmap_experts(model->expert_map, model->expert_map_size, model->expert_fd);
    rabbit_cache_free(model->cache);
    free(model);
}

rabbit_status rabbit_chat(RabbitModel *model) {
    if (!model) return RABBIT_ERR_ARGS;

    printf("\n[rabbit] Chat mode is a placeholder.\n");
    printf("[rabbit] Model name : %s\n", model->name);
    printf("[rabbit] Version    : %s\n", rabbit_version());
    printf("[rabbit] Next work  : implement forward pass + expert streaming.\n\n");

    printf("Type a message (or 'quit'):\n");

    char line[1024];
    while (1) {
        printf("> ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin))
            break;

        /* strip newline */
        size_t len = strlen(line);
        if (len && line[len-1] == '\n') line[len-1] = 0;

        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0)
            break;

        if (line[0] == 0)
            continue;

        printf("Rabbit (stub): I received \"%s\"\n", line);
        printf("               Real generation not implemented yet.\n");
    }

    return RABBIT_OK;
}
