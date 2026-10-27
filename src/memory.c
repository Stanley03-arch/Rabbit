#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifndef MADV_WILLNEED
#define MADV_WILLNEED 3
#endif
#ifndef MADV_DONTNEED
#define MADV_DONTNEED 4
#endif
#ifndef MADV_RANDOM
#define MADV_RANDOM 1
#endif

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

int rabbit_cache_pin(ExpertCache *c, int expert_id, int pinned) {
    if (!c || expert_id < 0) return -1;
    for (int i = 0; i < c->n_slots; i++) {
        if (c->slots[i].expert_id == expert_id) {
            c->slots[i].pinned = pinned ? 1 : 0;
            return 0;
        }
    }
    return -1;
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
    c->slots[victim].pinned = 0;
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

    /* Expert traffic is sparse: RANDOM reduces sequential readahead waste. */
    if (madvise(*base, *size, MADV_RANDOM) != 0) {
        /* non-fatal */
    }
    return 0;
}

void rabbit_munmap_file(void *base, size_t size, int fd) {
    if (base && base != MAP_FAILED)
        munmap(base, size);
    if (fd >= 0)
        close(fd);
}

int rabbit_expert_global_id(const RabbitModel *m, int layer, int local_expert_id) {
    if (!m || local_expert_id < 0) return -1;
    if (m->experts_shared)
        return local_expert_id;
    if (layer < 0 || layer >= m->n_layers) return -1;
    int epl = m->experts_per_layer > 0 ? m->experts_per_layer : m->num_experts;
    if (local_expert_id >= epl) return -1;
    return layer * epl + local_expert_id;
}

const void *rabbit_expert_ptr(const RabbitModel *m, int global_expert_id) {
    if (!m || !m->expert_map || global_expert_id < 0)
        return NULL;
    int nrec = m->n_expert_records > 0 ? m->n_expert_records : m->num_experts;
    if (global_expert_id >= nrec)
        return NULL;
    return (const uint8_t *)m->expert_map + (size_t)global_expert_id * m->expert_size;
}

const void *rabbit_expert_ptr_layer(const RabbitModel *m, int layer, int local_expert_id) {
    int gid = rabbit_expert_global_id(m, layer, local_expert_id);
    if (gid < 0) return NULL;
    return rabbit_expert_ptr(m, gid);
}

int rabbit_expert_prefetch(RabbitModel *m, int global_expert_id) {
    if (!m || !m->expert_map || global_expert_id < 0 || m->expert_size == 0)
        return -1;
    int nrec = m->n_expert_records > 0 ? m->n_expert_records : m->num_experts;
    if (global_expert_id >= nrec)
        return -1;

    void *p = (void *)((const uint8_t *)m->expert_map +
                       (size_t)global_expert_id * m->expert_size);
    if (madvise(p, m->expert_size, MADV_WILLNEED) != 0)
        return -1;
    return 0;
}

int rabbit_expert_release_hint(RabbitModel *m, int global_expert_id) {
    if (!m || !m->expert_map || global_expert_id < 0 || m->expert_size == 0)
        return -1;
    int nrec = m->n_expert_records > 0 ? m->n_expert_records : m->num_experts;
    if (global_expert_id >= nrec)
        return -1;

    void *p = (void *)((const uint8_t *)m->expert_map +
                       (size_t)global_expert_id * m->expert_size);
    if (madvise(p, m->expert_size, MADV_DONTNEED) != 0)
        return -1;
    return 0;
}
