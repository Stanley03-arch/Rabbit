#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct {
    int   expert_id;
    void *data;
    int   last_used;
    int   pinned;
} CacheSlot;

struct ExpertCache {
    CacheSlot slots[RABBIT_MAX_CACHE_SLOTS];
    int       clock;
    int       n_slots;
    size_t    expert_size;
    uint64_t  hits;
    uint64_t  misses;
    uint64_t  inserts;
};

ExpertCache *rabbit_cache_create(int n_slots) {
    if (n_slots <= 0 || n_slots > RABBIT_MAX_CACHE_SLOTS)
        n_slots = 4;

    ExpertCache *c = calloc(1, sizeof(ExpertCache));
    if (!c) return NULL;
    c->n_slots = n_slots;
    for (int i = 0; i < n_slots; i++)
        c->slots[i].expert_id = -1;
    return c;
}

void rabbit_cache_set_expert_size(ExpertCache *c, size_t size) {
    if (c) c->expert_size = size;
}

void rabbit_cache_free(ExpertCache *c) {
    if (!c) return;
    for (int i = 0; i < c->n_slots; i++)
        free(c->slots[i].data);
    free(c);
}

void rabbit_cache_stats(const ExpertCache *c, uint64_t *hits, uint64_t *misses, uint64_t *inserts) {
    if (hits) *hits = c ? c->hits : 0;
    if (misses) *misses = c ? c->misses : 0;
    if (inserts) *inserts = c ? c->inserts : 0;
}

void *rabbit_cache_get(ExpertCache *c, int expert_id) {
    if (!c) return NULL;
    for (int i = 0; i < c->n_slots; i++) {
        if (c->slots[i].expert_id == expert_id) {
            c->slots[i].last_used = ++c->clock;
            c->hits++;
            return c->slots[i].data;
        }
    }
    c->misses++;
    return NULL;
}

void *rabbit_cache_put(ExpertCache *c, int expert_id, const void *src) {
    if (!c || !src || c->expert_size == 0) return NULL;

    for (int i = 0; i < c->n_slots; i++) {
        if (c->slots[i].expert_id == expert_id) {
            c->slots[i].last_used = ++c->clock;
            return c->slots[i].data;
        }
    }

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
    if (victim < 0) return NULL;

    if (!c->slots[victim].data) {
        c->slots[victim].data = malloc(c->expert_size);
        if (!c->slots[victim].data) return NULL;
    }

    memcpy(c->slots[victim].data, src, c->expert_size);
    c->slots[victim].expert_id = expert_id;
    c->slots[victim].last_used = ++c->clock;
    c->inserts++;
    return c->slots[victim].data;
}

int rabbit_mmap_file(const char *path, void **base, size_t *size, int *fd) {
    *fd = open(path, O_RDONLY);
    if (*fd < 0) {
        perror("open");
        return -1;
    }

    struct stat st;
    if (fstat(*fd, &st) != 0) {
        perror("fstat");
        close(*fd);
        *fd = -1;
        return -1;
    }

    *size = (size_t)st.st_size;
    *base = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*base == MAP_FAILED) {
        perror("mmap");
        close(*fd);
        *fd = -1;
        *base = NULL;
        return -1;
    }
    return 0;
}

void rabbit_munmap_file(void *base, size_t size, int fd) {
    if (base && base != MAP_FAILED)
        munmap(base, size);
    if (fd >= 0)
        close(fd);
}

const void *rabbit_expert_ptr(const RabbitModel *m, int expert_id) {
    if (!m || !m->expert_map || expert_id < 0 || expert_id >= m->num_experts)
        return NULL;
    return (const uint8_t *)m->expert_map + (size_t)expert_id * m->expert_size;
}
