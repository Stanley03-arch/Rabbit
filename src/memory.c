#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/*
 * Memory management for Rabbit.
 *
 * Design goal for 4 GB machines:
 *   - Dense / shared weights stay in RAM
 *   - Experts are memory-mapped from disk
 *   - Only a tiny LRU of hot experts is kept in RAM
 */

typedef struct {
    int   expert_id;
    void *data;
    int   last_used;   /* simple clock for LRU */
} CacheSlot;

struct ExpertCache {
    CacheSlot slots[RABBIT_MAX_CACHE_SLOTS];
    int       clock;
    int       n_slots;
};

ExpertCache *rabbit_cache_create(int n_slots) {
    if (n_slots <= 0 || n_slots > RABBIT_MAX_CACHE_SLOTS)
        n_slots = RABBIT_MAX_CACHE_SLOTS;

    ExpertCache *c = calloc(1, sizeof(ExpertCache));
    if (!c) return NULL;

    c->n_slots = n_slots;
    c->clock = 0;

    for (int i = 0; i < n_slots; i++) {
        c->slots[i].expert_id = -1;
        c->slots[i].data = NULL;
        c->slots[i].last_used = 0;
    }
    return c;
}

void rabbit_cache_free(ExpertCache *c) {
    if (!c) return;
    for (int i = 0; i < c->n_slots; i++) {
        free(c->slots[i].data);
    }
    free(c);
}

/* Very simple mmap helper for expert file */
int rabbit_mmap_experts(const char *path, void **map, size_t *size, int *fd) {
    *fd = open(path, O_RDONLY);
    if (*fd < 0) {
        perror("open experts");
        return -1;
    }

    struct stat st;
    if (fstat(*fd, &st) < 0) {
        perror("fstat");
        close(*fd);
        return -1;
    }

    *size = (size_t)st.st_size;
    *map = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*map == MAP_FAILED) {
        perror("mmap");
        close(*fd);
        return -1;
    }

    return 0;
}

void rabbit_munmap_experts(void *map, size_t size, int fd) {
    if (map && map != MAP_FAILED)
        munmap(map, size);
    if (fd >= 0)
        close(fd);
}
