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
    void *data;          /* copy in RAM (optional) */
    int   last_used;     /* simple clock for LRU */
    int   pinned;        /* never evict if set */
} CacheSlot;

struct ExpertCache {
    CacheSlot slots[RABBIT_MAX_CACHE_SLOTS];
    int       clock;
    int       n_slots;
    size_t    expert_size;
};

ExpertCache *rabbit_cache_create(int n_slots) {
    if (n_slots <= 0 || n_slots > RABBIT_MAX_CACHE_SLOTS)
        n_slots = 4;   /* default very small for 4GB target */

    ExpertCache *c = calloc(1, sizeof(ExpertCache));
    if (!c) return NULL;

    c->n_slots = n_slots;
    c->clock = 0;
    c->expert_size = 0;

    for (int i = 0; i < n_slots; i++) {
        c->slots[i].expert_id = -1;
        c->slots[i].data = NULL;
        c->slots[i].last_used = 0;
        c->slots[i].pinned = 0;
    }
    return c;
}

void rabbit_cache_set_expert_size(ExpertCache *c, size_t size) {
    if (c) c->expert_size = size;
}

void rabbit_cache_free(ExpertCache *c) {
    if (!c) return;
    for (int i = 0; i < c->n_slots; i++) {
        free(c->slots[i].data);
    }
    free(c);
}

/* Find expert in cache. Returns pointer or NULL. */
void *rabbit_cache_get(ExpertCache *c, int expert_id) {
    if (!c) return NULL;
    for (int i = 0; i < c->n_slots; i++) {
        if (c->slots[i].expert_id == expert_id) {
            c->slots[i].last_used = ++c->clock;
            return c->slots[i].data;
        }
    }
    return NULL;
}

/* Insert or replace using simple LRU. Returns the RAM pointer. */
void *rabbit_cache_put(ExpertCache *c, int expert_id, const void *src) {
    if (!c || !src || c->expert_size == 0) return NULL;

    /* Already present? */
    for (int i = 0; i < c->n_slots; i++) {
        if (c->slots[i].expert_id == expert_id) {
            c->slots[i].last_used = ++c->clock;
            return c->slots[i].data;
        }
    }

    /* Find free or LRU slot */
    int victim = -1;
    int oldest = 0x7fffffff;

    for (int i = 0; i < c->n_slots; i++) {
        if (c->slots[i].expert_id < 0) {
            victim = i;
            break;
        }
        if (!c->slots[i].pinned && c->slots[i].last_used < oldest) {
            oldest = c->slots[i].last_used;
            victim = i;
        }
    }

    if (victim < 0) return NULL;  /* all pinned */

    /* Allocate if needed */
    if (!c->slots[victim].data) {
        c->slots[victim].data = malloc(c->expert_size);
        if (!c->slots[victim].data) return NULL;
    }

    memcpy(c->slots[victim].data, src, c->expert_size);
    c->slots[victim].expert_id = expert_id;
    c->slots[victim].last_used = ++c->clock;

    return c->slots[victim].data;
}

/* ------------------------------------------------------------
 * mmap helpers for the expert section
 * ------------------------------------------------------------ */

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
        *fd = -1;
        return -1;
    }

    *size = (size_t)st.st_size;
    *map = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*map == MAP_FAILED) {
        perror("mmap");
        close(*fd);
        *fd = -1;
        *map = NULL;
        return -1;
    }

    return 0;
}

/* Map only the expert region (offset + length) */
int rabbit_mmap_experts_region(const char *path, off_t offset, size_t length,
                               void **map, int *fd) {
    *fd = open(path, O_RDONLY);
    if (*fd < 0) {
        perror("open experts");
        return -1;
    }

    *map = mmap(NULL, length, PROT_READ, MAP_PRIVATE, *fd, offset);
    if (*map == MAP_FAILED) {
        perror("mmap region");
        close(*fd);
        *fd = -1;
        *map = NULL;
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

/* Get pointer to a specific expert inside the mapped region */
const void *rabbit_expert_ptr(const RabbitModel *m, int expert_id) {
    if (!m || !m->expert_map || expert_id < 0 || expert_id >= m->num_experts)
        return NULL;

    return (const uint8_t *)m->expert_map + (size_t)expert_id * m->expert_size;
}
