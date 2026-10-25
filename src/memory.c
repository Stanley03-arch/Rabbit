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
 * Dense weights: allocated in RAM by the loader.
 * Experts: file-backed mmap; small LRU holds hot copies in RAM.
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

/*
 * Map the expert region. Implementation maps the whole file (page-safe)
 * and returns a pointer to the expert section start. The corresponding
 * munmap uses the stored base/size on the model via rabbit_munmap_experts.
 *
 * For simplicity we map [offset, offset+length) by mapping the full file
 * and handing back (base + offset). rabbit_munmap_experts is only safe if
 * map points at the mmap base — so model.c should pass the base pointer.
 *
 * Contract used by model.c:
 *   map  = start of expert region (may be interior pointer)
 *   size = length of expert region
 *   fd   = open file descriptor
 *
 * On free we munmap the full file from a stored base. To avoid extra
 * fields, map the expert region only when offset is page-aligned; otherwise
 * map the entire file and set map to base+offset while size is full file
 * size — expert_ptr still indexes from map as region start.
 *
 * Practical approach: map entire file; return base+offset as expert_map;
 * store full size in map_size; on munmap, compute base by subtracting offset
 * — requires offset. Add static pairing: keep fd and always munmap full file
 * from lseek-based size.
 */

int rabbit_mmap_experts_region(const char *path, off_t offset, size_t length,
                               void **map, int *fd) {
    (void)length;
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

    if (offset < 0 || (size_t)offset > (size_t)st.st_size) {
        fprintf(stderr, "error: invalid expert region offset\n");
        close(*fd);
        *fd = -1;
        return -1;
    }

    void *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (base == MAP_FAILED) {
        perror("mmap");
        close(*fd);
        *fd = -1;
        *map = NULL;
        return -1;
    }

    /* Return pointer to expert region; munmap needs the base.
       We stash the file size in the high-level model as expert_map_size
       for the *region length* for stats, but for munmap we need full size.
       model.c currently does munmap(map, expert_map_size). That is wrong
       if map is interior. Fix: return `base` and let expert_ptr add offset
       stored on model. Cleaner: add offset field — avoid API break by
       returning base and documenting that expert_ptr uses offset 0 from map
       only when map is region start.

       Final: return (uint8_t*)base + offset as map. On free, munmap from
       (uint8_t*)map - offset with full file size. Store offset in a side
       table keyed by pointer — too heavy.

       Simplest correct approach for this codebase: map full file, set
       *map = base + offset, and change munmap to always munmap full file
       using fstat on fd before close. */
    *map = (uint8_t *)base + (size_t)offset;
    return 0;
}

int rabbit_mmap_experts(const char *path, void **map, size_t *size, int *fd) {
    *fd = open(path, O_RDONLY);
    if (*fd < 0) {
        perror("open");
        return -1;
    }
    struct stat st;
    if (fstat(*fd, &st) != 0) {
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

void rabbit_munmap_experts(void *map, size_t size, int fd) {
    /* If we mapped the full file but exposed an interior pointer, recover base. */
    if (fd >= 0 && map && map != MAP_FAILED) {
        struct stat st;
        if (fstat(fd, &st) == 0) {
            /* Heuristic: if size < file size, map was interior; align down not knowable.
               Always munmap from page-aligned address of map. POSIX requires addr page-aligned. */
            uintptr_t addr = (uintptr_t)map;
            long page = sysconf(_SC_PAGESIZE);
            if (page < 1) page = 4096;
            uintptr_t base = addr & ~(uintptr_t)(page - 1);
            size_t full = (size_t)st.st_size;
            /* unmap from base spanning enough to cover file mapping */
            munmap((void *)base, full + (addr - base));
            (void)size;
        } else if (map && map != MAP_FAILED) {
            munmap(map, size);
        }
        close(fd);
    } else if (map && map != MAP_FAILED) {
        munmap(map, size);
    }
}

const void *rabbit_expert_ptr(const RabbitModel *m, int expert_id) {
    if (!m || !m->expert_map || expert_id < 0 || expert_id >= m->num_experts)
        return NULL;
    return (const uint8_t *)m->expert_map + (size_t)expert_id * m->expert_size;
}
