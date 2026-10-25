#include "rabbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/*
 * Minimal byte-level tokenizer for the synthetic test model.
 * Maps each byte 0..255 to a token id. This is enough to prove
 * the generation loop before we add real SentencePiece / tiktoken support.
 */

#define RABBIT_VOCAB 256

typedef struct {
    int vocab_size;
} RabbitTokenizer;

static RabbitTokenizer g_tok = { .vocab_size = RABBIT_VOCAB };

int rabbit_tokenizer_vocab(void) {
    return g_tok.vocab_size;
}

/* Encode a UTF-8 / ASCII string into token ids (one byte = one token) */
int rabbit_tokenize(const char *text, int *out_ids, int max_ids) {
    if (!text || !out_ids || max_ids <= 0) return 0;

    int n = 0;
    const unsigned char *p = (const unsigned char *)text;
    while (*p && n < max_ids) {
        out_ids[n++] = (int)(*p);
        p++;
    }
    return n;
}

/* Decode a single token id back to a byte (as a short string) */
void rabbit_detokenize(int id, char *out, int out_size) {
    if (!out || out_size < 2) return;
    if (id < 0 || id >= RABBIT_VOCAB) {
        out[0] = '?';
        out[1] = 0;
        return;
    }
    out[0] = (char)id;
    out[1] = 0;
}

/* Decode a sequence of ids into a C string */
void rabbit_detokenize_seq(const int *ids, int n, char *out, int out_size) {
    if (!out || out_size <= 0) return;
    int pos = 0;
    for (int i = 0; i < n && pos < out_size - 1; i++) {
        int id = ids[i];
        if (id >= 0 && id < RABBIT_VOCAB)
            out[pos++] = (char)id;
        else
            out[pos++] = '?';
    }
    out[pos] = 0;
}
