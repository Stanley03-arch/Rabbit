#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

/* Small utility helpers */

void rabbit_die(const char *msg) {
    fprintf(stderr, "Fatal: %s\n", msg);
    exit(1);
}

void rabbit_dief(const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "Fatal: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

void *rabbit_xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) rabbit_die("out of memory");
    return p;
}

void *rabbit_xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) rabbit_die("out of memory");
    return p;
}

void *rabbit_xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n);
    if (!p && n > 0) rabbit_die("out of memory");
    return p;
}

char *rabbit_xstrdup(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s) + 1;
    char *p = (char *)rabbit_xmalloc(len);
    memcpy(p, s, len);
    return p;
}
