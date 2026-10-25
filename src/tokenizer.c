#define _POSIX_C_SOURCE 200809L

#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <limits.h>

#define RABBIT_BYTE_VOCAB 256
#define RABBIT_MAX_WORD_VOCAB 4096
#define RABBIT_BPE_MAX_VOCAB 262144
#define RABBIT_BPE_MAX_MERGES 262144
#define RABBIT_BPE_MAX_TOKEN_LEN 128
#define RABBIT_BPE_MAX_PIECES 4096

typedef enum {
    TOK_BYTE = 0,
    TOK_WORD = 1,
    TOK_BPE  = 2
} TokMode;

typedef struct {
    char *str;
    int   len;
} BpeToken;

typedef struct {
    int left;
    int right;
    int rank;
} BpeMerge;

typedef struct {
    TokMode   mode;
    int       vocab_size;

    /* word mode */
    char    **words;
    int       n_words;

    /* BPE mode */
    BpeToken *vocab;
    int       n_vocab;
    BpeMerge *merges;
    int       n_merges;
    int      *pair_table;
    int       pair_table_size;
    int       unk_id;
    int       bos_id;
    int       eos_id;
    int       pad_id;
} RabbitTokenizer;

static RabbitTokenizer g_tok = {
    .mode = TOK_BYTE,
    .vocab_size = RABBIT_BYTE_VOCAB,
    .words = NULL,
    .n_words = 0,
    .vocab = NULL,
    .n_vocab = 0,
    .merges = NULL,
    .n_merges = 0,
    .pair_table = NULL,
    .pair_table_size = 0,
    .unk_id = 0,
    .bos_id = 1,
    .eos_id = 2,
    .pad_id = 3
};

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static void free_bpe(void) {
    if (g_tok.vocab) {
        for (int i = 0; i < g_tok.n_vocab; i++)
            free(g_tok.vocab[i].str);
        free(g_tok.vocab);
        g_tok.vocab = NULL;
    }
    free(g_tok.merges);
    g_tok.merges = NULL;
    free(g_tok.pair_table);
    g_tok.pair_table = NULL;
    g_tok.n_vocab = 0;
    g_tok.n_merges = 0;
    g_tok.pair_table_size = 0;
}

static void free_words(void) {
    for (int i = 0; i < g_tok.n_words; i++)
        free(g_tok.words[i]);
    free(g_tok.words);
    g_tok.words = NULL;
    g_tok.n_words = 0;
}

static uint32_t pair_hash(int left, int right) {
    uint32_t h = (uint32_t)left * 2654435761u ^ (uint32_t)right * 2246822519u;
    return h;
}

static int pair_lookup(int left, int right) {
    if (!g_tok.pair_table || g_tok.pair_table_size <= 0) return -1;
    uint32_t h = pair_hash(left, right) % (uint32_t)g_tok.pair_table_size;
    for (int probe = 0; probe < g_tok.pair_table_size; probe++) {
        int idx = (int)((h + (uint32_t)probe) % (uint32_t)g_tok.pair_table_size);
        int *e = g_tok.pair_table + idx * 3;
        if (e[0] == -1) return -1;
        if (e[0] == left && e[1] == right) return e[2];
    }
    return -1;
}

static int build_pair_table(void) {
    int size = 1;
    while (size < g_tok.n_merges * 2) size <<= 1;
    if (size < 1024) size = 1024;

    int *table = (int *)malloc(sizeof(int) * (size_t)size * 3);
    if (!table) return -1;
    for (int i = 0; i < size * 3; i++) table[i] = -1;

    for (int m = 0; m < g_tok.n_merges; m++) {
        int left = g_tok.merges[m].left;
        int right = g_tok.merges[m].right;
        int rank = g_tok.merges[m].rank;
        uint32_t h = pair_hash(left, right) % (uint32_t)size;
        for (int probe = 0; probe < size; probe++) {
            int idx = (int)((h + (uint32_t)probe) % (uint32_t)size);
            int *e = table + idx * 3;
            if (e[0] == -1) {
                e[0] = left;
                e[1] = right;
                e[2] = rank;
                break;
            }
        }
    }
    free(g_tok.pair_table);
    g_tok.pair_table = table;
    g_tok.pair_table_size = size;
    return 0;
}

int rabbit_tokenizer_vocab(void) {
    return g_tok.vocab_size;
}

void rabbit_tokenizer_set_byte_mode(void) {
    free_bpe();
    free_words();
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

    free_bpe();
    free_words();
    g_tok.words = words;
    g_tok.n_words = n;
    g_tok.mode = TOK_WORD;
    g_tok.vocab_size = n + 4;
    return 0;
}

/*
 * Compact Rabbit BPE format (.tok):
 *   magic[8] = "RBTTOK01"
 *   u32 version = 1
 *   u32 vocab_size
 *   u32 n_merges
 *   i32 unk_id, bos_id, eos_id, pad_id
 *   for each vocab token: u16 len + bytes
 *   for each merge: u32 left_id, u32 right_id, u32 rank
 */
int rabbit_tokenizer_load_bpe(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;

    char magic[8];
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "RBTTOK01", 8) != 0) {
        fclose(fp);
        return -1;
    }

    uint32_t version = 0, vocab_size = 0, n_merges = 0;
    int32_t unk_id = 0, bos_id = 1, eos_id = 2, pad_id = 3;
    if (fread(&version, 4, 1, fp) != 1 ||
        fread(&vocab_size, 4, 1, fp) != 1 ||
        fread(&n_merges, 4, 1, fp) != 1 ||
        fread(&unk_id, 4, 1, fp) != 1 ||
        fread(&bos_id, 4, 1, fp) != 1 ||
        fread(&eos_id, 4, 1, fp) != 1 ||
        fread(&pad_id, 4, 1, fp) != 1) {
        fclose(fp);
        return -1;
    }

    if (version != 1 || vocab_size == 0 || vocab_size > RABBIT_BPE_MAX_VOCAB ||
        n_merges > RABBIT_BPE_MAX_MERGES) {
        fclose(fp);
        return -1;
    }

    BpeToken *vocab = (BpeToken *)calloc(vocab_size, sizeof(BpeToken));
    if (!vocab) {
        fclose(fp);
        return -1;
    }

    for (uint32_t i = 0; i < vocab_size; i++) {
        uint16_t len = 0;
        if (fread(&len, 2, 1, fp) != 1 || len > RABBIT_BPE_MAX_TOKEN_LEN) {
            for (uint32_t j = 0; j < i; j++) free(vocab[j].str);
            free(vocab);
            fclose(fp);
            return -1;
        }
        char *s = (char *)malloc((size_t)len + 1);
        if (!s || (len && fread(s, 1, len, fp) != len)) {
            free(s);
            for (uint32_t j = 0; j < i; j++) free(vocab[j].str);
            free(vocab);
            fclose(fp);
            return -1;
        }
        s[len] = 0;
        vocab[i].str = s;
        vocab[i].len = (int)len;
    }

    BpeMerge *merges = NULL;
    if (n_merges > 0) {
        merges = (BpeMerge *)calloc(n_merges, sizeof(BpeMerge));
        if (!merges) {
            for (uint32_t i = 0; i < vocab_size; i++) free(vocab[i].str);
            free(vocab);
            fclose(fp);
            return -1;
        }
        for (uint32_t i = 0; i < n_merges; i++) {
            uint32_t left = 0, right = 0, rank = 0;
            if (fread(&left, 4, 1, fp) != 1 ||
                fread(&right, 4, 1, fp) != 1 ||
                fread(&rank, 4, 1, fp) != 1) {
                free(merges);
                for (uint32_t j = 0; j < vocab_size; j++) free(vocab[j].str);
                free(vocab);
                fclose(fp);
                return -1;
            }
            merges[i].left = (int)left;
            merges[i].right = (int)right;
            merges[i].rank = (int)rank;
        }
    }
    fclose(fp);

    free_bpe();
    free_words();

    g_tok.vocab = vocab;
    g_tok.n_vocab = (int)vocab_size;
    g_tok.merges = merges;
    g_tok.n_merges = (int)n_merges;
    g_tok.unk_id = unk_id;
    g_tok.bos_id = bos_id;
    g_tok.eos_id = eos_id;
    g_tok.pad_id = pad_id;
    g_tok.mode = TOK_BPE;
    g_tok.vocab_size = (int)vocab_size;

    if (build_pair_table() != 0) {
        free_bpe();
        g_tok.mode = TOK_BYTE;
        g_tok.vocab_size = RABBIT_BYTE_VOCAB;
        return -1;
    }
    return 0;
}

static int find_token_bytes(const char *s, int len) {
    for (int i = 0; i < g_tok.n_vocab; i++) {
        if (g_tok.vocab[i].len == len &&
            memcmp(g_tok.vocab[i].str, s, (size_t)len) == 0)
            return i;
    }
    return -1;
}

static int bpe_merge_ids(int *ids, int n) {
    if (n <= 1) return n;

    while (1) {
        int best_rank = INT_MAX;
        int best_i = -1;
        for (int i = 0; i < n - 1; i++) {
            int rank = pair_lookup(ids[i], ids[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }
        if (best_i < 0) break;

        int left = ids[best_i];
        int right = ids[best_i + 1];
        if (left < 0 || left >= g_tok.n_vocab || right < 0 || right >= g_tok.n_vocab)
            break;

        int llen = g_tok.vocab[left].len;
        int rlen = g_tok.vocab[right].len;
        char buf[RABBIT_BPE_MAX_TOKEN_LEN * 2 + 1];
        if (llen + rlen > RABBIT_BPE_MAX_TOKEN_LEN * 2) break;
        memcpy(buf, g_tok.vocab[left].str, (size_t)llen);
        memcpy(buf + llen, g_tok.vocab[right].str, (size_t)rlen);
        int merged_id = find_token_bytes(buf, llen + rlen);
        if (merged_id < 0) break;

        ids[best_i] = merged_id;
        memmove(ids + best_i + 1, ids + best_i + 2,
                sizeof(int) * (size_t)(n - best_i - 2));
        n--;
    }
    return n;
}

static int tokenize_bpe(const char *text, int *out_ids, int max_ids) {
    if (!text || !out_ids || max_ids <= 0) return 0;

    int pieces[RABBIT_BPE_MAX_PIECES];
    int np = 0;
    const unsigned char *p = (const unsigned char *)text;

    while (*p && np < RABBIT_BPE_MAX_PIECES) {
        char b[2] = { (char)(*p), 0 };
        int id = find_token_bytes(b, 1);
        if (id < 0) id = g_tok.unk_id;
        pieces[np++] = id;
        p++;
    }

    np = bpe_merge_ids(pieces, np);

    int n = 0;
    for (int i = 0; i < np && n < max_ids; i++)
        out_ids[n++] = pieces[i];
    return n;
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

    if (g_tok.mode == TOK_BPE)
        return tokenize_bpe(text, out_ids, max_ids);

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
        if (id < 0) id = g_tok.n_words;
        out_ids[n++] = id;
    }
    return n;
}

void rabbit_detokenize(int id, char *out, int out_size) {
    if (!out || out_size < 2) return;

    if (g_tok.mode == TOK_BPE) {
        if (id >= 0 && id < g_tok.n_vocab && g_tok.vocab[id].str) {
            int len = g_tok.vocab[id].len;
            if (len >= out_size) len = out_size - 1;
            memcpy(out, g_tok.vocab[id].str, (size_t)len);
            out[len] = 0;
        } else {
            strncpy(out, "<unk>", (size_t)out_size - 1);
            out[out_size - 1] = 0;
        }
        return;
    }

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
        char tmp[RABBIT_BPE_MAX_TOKEN_LEN + 8];
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
