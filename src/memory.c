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
 * Dense / shared weights stay in RAM.
 * Experts are memory-mapped; a small LRU keeps hot experts resident.
 */

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

    /* stats */
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
    c->clock = 0;
    c->expert_size = 0;
    c->hits = c->misses = c->inserts = 0;

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
    for (int i = 0; i < c->n_slots; i++)
        free(c->slots[i].data);
    free(c);
}

void rabbit_cache_stats(const ExpertCache *c, uint64_t *hits, uint64_t *misses, uint64_t *inserts) {
    if (!c) {
        if (hits) *hits = 0;
        if (misses) *misses = 0;
        if (inserts) *inserts = 0;
        return;
    }
    if (hits) *hits = c->hits;
    if (misses) *misses = c->misses;
    if (inserts) *inserts = c->inserts;
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

int rabbit_mmap_experts(const char *path, void **map, size_t *size, int *fd) {
    *fd = open(path, O_RDONLY);
    if (*fd < 0) {
        perror("open");
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

int rabbit_mmap_experts_region(const char *path, off_t offset, size_t length,
                               void **map, int *fd) {
    *fd = open(path, O_RDONLY);
    if (*fd < 0) {
        perror("open");
        return -1;
    }

    /* page-align offset for portability */
    long page = sysconf(_SC_PAGESIZE);
    if (page < 1) page = 4096;
    off_t aligned = offset & ~(off_t)(page - 1);
    size_t delta = (size_t)(offset - aligned);
    size_t map_len = length + delta;

    void *base = mmap(NULL, map_len, PROT_READ, MAP_PRIVATE, *fd, aligned);
    if (base == MAP_FAILED) {
        perror("mmap");
        close(*fd);
        *fd = -1;
        *map = NULL;
        return -1;
    }

    *map = (uint8_t *)base + delta;
    /* store base/len in a way free can use: we keep fd; munmap needs base.
       For simplicity, munmap from the returned pointer only works if delta==0.
       So prefer aligned path by adjusting: always return base and track delta in model.
       Here length passed to munmap from model is expert_map_size; we map full region
       from file start of expert section which create_test uses as aligned enough.
       If offset is not page-aligned, store unaligned pointer — on Linux mmap requires
       page-aligned offset. Fix: map from aligned and expose unaligned pointer; on free
       we need original base. Simplify: require caller offset page-aligned OR map whole file. */

    /* Robust approach: map entire file then pointer arithmetic in rabbit_expert_ptr.
       For region map, if offset not aligned, fall back to full-file map. */
    if (delta != 0) {
        munmap(base, map_len);
        struct stat st;
        if (fstat(*fd, &st) != 0) {
            close(*fd);
            *fd = -1;
            *map = NULL;
            return -1;
        }
        void *whole = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, *fd, 0);
        if (whole == MAP_FAILED) {
            perror("mmap");
            close(*fd);
            *fd = -1;
            *map = NULL;
            return -1;
        }
        *map = (uint8_t *)whole + (size_t)offset;
        /* Note: rabbit_munmap will munmap from *map with length — incorrect for subpointer.
           model.c stores expert_map and expert_map_size as remaining size.
           We'll document that region map uses full-file map when unaligned and
           rabbit_munmap_experts must receive the base. For now, pass whole as map
           only when aligned; when unaligned store whole and let expert_ptr add offset.

           Simplest fix used below: always full-file map for region API when needed. */
        (void)length;
        /* Overwrite: model expects map to start at expert region. Store adjusted pointer;
           munmap in free uses expert_map_size which is remaining — WRONG for full map.
           Update model.c path: use full file map and expert_ptr already uses expert_id * size
           from start of expert region. model.c sets map to expert region start.

           Keep: return (uint8_t*)whole + offset; on free, munmap((uint8_t*)map - offset, filesize).
           That requires storing offset. For v0.2.2, if unaligned, fail clearly. */
        munmap(whole, (size_t)st.st_size);
        fprintf(stderr, "error: expert region offset is not page-aligned (%ld)\n", (long)offset);
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

const void *rabbit_expert_ptr(const RabbitModel *m, int expert_id) {
    if (!m || !m->expert_map || expert_id < 0 || expert_id >= m->num_experts)
        return NULL;
    return (const uint8_t *)m->expert_map + (size_t)expert_id * m->expert_size;
}
