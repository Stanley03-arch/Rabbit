#define _POSIX_C_SOURCE 200809L

#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define RABBIT_BYTE_VOCAB 256
#define RABBIT_MAX_WORD_VOCAB 4096

typedef enum {
    TOK_BYTE = 0,
    TOK_WORD = 1
} TokMode;

typedef struct {
    TokMode mode;
    int     vocab_size;
    char  **words;
    int     n_words;
} RabbitTokenizer;

static RabbitTokenizer g_tok = {
    .mode = TOK_BYTE,
    .vocab_size = RABBIT_BYTE_VOCAB,
    .words = NULL,
    .n_words = 0
};

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

int rabbit_tokenizer_vocab(void) {
    return g_tok.vocab_size;
}

void rabbit_tokenizer_set_byte_mode(void) {
    g_tok.mode = TOK_BYTE;
    g_tok.vocab_size = RABBIT_BYTE_VOCAB;
}

int rabbit_tokenizer_load_vocab(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;

    char **words = (char **)calloc(RABBIT_MAX_WORD_VOCAB, sizeof(char *));
    if (!words) {
        fclose(fp);
        return -1;
    }

    char line[256];
    int n = 0;
    while (fgets(line, sizeof(line), fp) && n < RABBIT_MAX_WORD_VOCAB) {
        size_t len = strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = 0;
        if (len == 0) continue;

        words[n] = xstrdup(line);
        if (!words[n]) break;
        n++;
    }
    fclose(fp);

    if (n == 0) {
        free(words);
        return -1;
    }

    for (int i = 0; i < g_tok.n_words; i++)
        free(g_tok.words[i]);
    free(g_tok.words);

    g_tok.words = words;
    g_tok.n_words = n;
    g_tok.mode = TOK_WORD;
    g_tok.vocab_size = n + 4;
    return 0;
}

static int find_word(const char *w) {
    for (int i = 0; i < g_tok.n_words; i++) {
        if (strcmp(g_tok.words[i], w) == 0)
            return i;
    }
    return -1;
}

int rabbit_tokenize(const char *text, int *out_ids, int max_ids) {
    if (!text || !out_ids || max_ids <= 0) return 0;

    if (g_tok.mode == TOK_BYTE) {
        int n = 0;
        const unsigned char *p = (const unsigned char *)text;
        while (*p && n < max_ids) {
            out_ids[n++] = (int)(*p);
            p++;
        }
        return n;
    }

    int n = 0;
    const char *p = text;
    char buf[128];

    while (*p && n < max_ids) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        if (ispunct((unsigned char)*p)) {
            buf[0] = *p++;
            buf[1] = 0;
        } else {
            int i = 0;
            while (*p && !isspace((unsigned char)*p) && !ispunct((unsigned char)*p) && i < 126)
                buf[i++] = *p++;
            buf[i] = 0;
        }

        int id = find_word(buf);
        if (id < 0) id = g_tok.n_words; /* UNK */
        out_ids[n++] = id;
    }
    return n;
}

void rabbit_detokenize(int id, char *out, int out_size) {
    if (!out || out_size < 2) return;

    if (g_tok.mode == TOK_BYTE) {
        if (id < 0 || id >= RABBIT_BYTE_VOCAB) {
            out[0] = '?';
            out[1] = 0;
            return;
        }
        out[0] = (char)id;
        out[1] = 0;
        return;
    }

    if (id >= 0 && id < g_tok.n_words) {
        strncpy(out, g_tok.words[id], (size_t)out_size - 1);
        out[out_size - 1] = 0;
    } else {
        strncpy(out, "<unk>", (size_t)out_size - 1);
        out[out_size - 1] = 0;
    }
}

void rabbit_detokenize_seq(const int *ids, int n, char *out, int out_size) {
    if (!out || out_size <= 0) return;
    int pos = 0;
    out[0] = 0;

    for (int i = 0; i < n && pos < out_size - 1; i++) {
        char tmp[128];
        rabbit_detokenize(ids[i], tmp, (int)sizeof(tmp));
        int len = (int)strlen(tmp);
        if (pos + len + 1 >= out_size) break;

        if (g_tok.mode == TOK_WORD && i > 0 && tmp[0] && !ispunct((unsigned char)tmp[0]))
            out[pos++] = ' ';

        memcpy(out + pos, tmp, (size_t)len);
        pos += len;
        out[pos] = 0;
    }
}
