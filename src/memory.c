#define _POSIX_C_SOURCE 200809L
#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

typedef struct {
    int   expert_id;   /* -1 = empty */
    void *data;
    int   last_used;   /* LRU clock */
    int   pinned;      /* never evict when set */
} CacheSlot;

struct ExpertCache {
    CacheSlot slots[RABBIT_MAX_CACHE_SLOTS];
    int       clock;
    int       n_slots;
    size_t    expert_size;
    uint64_t  hits;
    uint64_t  misses;
    uint64_t  inserts;
    uint64_t  evictions;
    uint64_t  bytes_copied;
};

ExpertCache *rabbit_cache_create(int n_slots) {
    if (n_slots <= 0)
        n_slots = 4;
    if (n_slots > RABBIT_MAX_CACHE_SLOTS)
        n_slots = RABBIT_MAX_CACHE_SLOTS;

    ExpertCache *c = calloc(1, sizeof(ExpertCache));
    if (!c) return NULL;
    c->n_slots = n_slots;
    for (int i = 0; i < n_slots; i++)
        c->slots[i].expert_id = -1;
    return c;
}

void rabbit_cache_set_expert_size(ExpertCache *c, size_t size) {
    if (!c) return;
    c->expert_size = size;
    /* Pre-allocate all slot buffers so generation never hits malloc. */
    if (size == 0) return;
    for (int i = 0; i < c->n_slots; i++) {
        if (!c->slots[i].data) {
            c->slots[i].data = malloc(size);
            if (!c->slots[i].data) {
                /* Best-effort: leave remaining slots NULL; put() will retry. */
                break;
            }
        }
    }
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

void rabbit_cache_stats_ex(const ExpertCache *c, uint64_t *hits, uint64_t *misses,
                           uint64_t *inserts, uint64_t *evictions, uint64_t *bytes_copied,
                           int *n_slots, int *n_pinned) {
    if (hits) *hits = c ? c->hits : 0;
    if (misses) *misses = c ? c->misses : 0;
    if (inserts) *inserts = c ? c->inserts : 0;
    if (evictions) *evictions = c ? c->evictions : 0;
    if (bytes_copied) *bytes_copied = c ? c->bytes_copied : 0;
    if (n_slots) *n_slots = c ? c->n_slots : 0;
    if (n_pinned) {
        int p = 0;
        if (c) {
            for (int i = 0; i < c->n_slots; i++)
                if (c->slots[i].pinned && c->slots[i].expert_id >= 0)
                    p++;
        }
        *n_pinned = p;
    }
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

static int find_victim(ExpertCache *c) {
    /* Prefer empty slots. */
    for (int i = 0; i < c->n_slots; i++) {
        if (c->slots[i].expert_id < 0)
            return i;
    }
    /* LRU among unpinned slots. */
    int victim = -1;
    int oldest = 0x7fffffff;
    for (int i = 0; i < c->n_slots; i++) {
        if (c->slots[i].pinned)
            continue;
        if (c->slots[i].last_used < oldest) {
            oldest = c->slots[i].last_used;
            victim = i;
        }
    }
    return victim;
}

void *rabbit_cache_put(ExpertCache *c, int expert_id, const void *src) {
    if (!c || !src || c->expert_size == 0) return NULL;

    /* Already resident? */
    for (int i = 0; i < c->n_slots; i++) {
        if (c->slots[i].expert_id == expert_id) {
            c->slots[i].last_used = ++c->clock;
            return c->slots[i].data;
        }
    }

    int victim = find_victim(c);
    if (victim < 0)
        return NULL; /* all slots pinned and full */

    if (c->slots[victim].expert_id >= 0)
        c->evictions++;

    if (!c->slots[victim].data) {
        c->slots[victim].data = malloc(c->expert_size);
        if (!c->slots[victim].data) return NULL;
    }

    memcpy(c->slots[victim].data, src, c->expert_size);
    c->slots[victim].expert_id = expert_id;
    c->slots[victim].last_used = ++c->clock;
    c->slots[victim].pinned = 0;
    c->inserts++;
    c->bytes_copied += c->expert_size;
    return c->slots[victim].data;
}

int rabbit_cache_pin(ExpertCache *c, int expert_id, const void *src_or_null) {
    if (!c || expert_id < 0) return -1;

    /* Already present — just pin. */
    for (int i = 0; i < c->n_slots; i++) {
        if (c->slots[i].expert_id == expert_id) {
            c->slots[i].pinned = 1;
            c->slots[i].last_used = ++c->clock;
            return 0;
        }
    }

    if (!src_or_null || c->expert_size == 0)
        return -1;

    int victim = find_victim(c);
    if (victim < 0)
        return -1;

    if (c->slots[victim].expert_id >= 0)
        c->evictions++;

    if (!c->slots[victim].data) {
        c->slots[victim].data = malloc(c->expert_size);
        if (!c->slots[victim].data) return -1;
    }

    memcpy(c->slots[victim].data, src_or_null, c->expert_size);
    c->slots[victim].expert_id = expert_id;
    c->slots[victim].last_used = ++c->clock;
    c->slots[victim].pinned = 1;
    c->inserts++;
    c->bytes_copied += c->expert_size;
    return 0;
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

int rabbit_expert_pread(const RabbitModel *m, int global_expert_id, void *dst) {
    if (!m || !dst || global_expert_id < 0 || m->expert_size == 0)
        return -1;
    if (m->expert_fd < 0)
        return -1;
    int nrec = m->n_expert_records > 0 ? m->n_expert_records : m->num_experts;
    if (global_expert_id >= nrec)
        return -1;
    off_t off = m->expert_file_offset + (off_t)global_expert_id * (off_t)m->expert_size;
    size_t left = m->expert_size;
    uint8_t *p = (uint8_t *)dst;
    while (left > 0) {
        ssize_t n = pread(m->expert_fd, p, left, off);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0)
            return -1; /* EOF / short */
        p += n;
        off += n;
        left -= (size_t)n;
    }
    return 0;
}
