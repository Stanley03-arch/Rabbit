#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>

/* Small utility helpers */

void rabbit_die(const char *msg) {
    fprintf(stderr, "Fatal: %s\n", msg);
    exit(1);
}

void *rabbit_xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) rabbit_die("out of memory");
    return p;
}
