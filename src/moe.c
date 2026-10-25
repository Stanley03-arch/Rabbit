#include "rabbit.h"
#include <stdio.h>

/*
 * Mixture-of-Experts routing + expert streaming logic.
 *
 * This is the heart of Rabbit (and Colibrì):
 *   1. Run router on the current hidden state
 *   2. Select top-k experts
 *   3. For each selected expert:
 *        - check tiny RAM cache
 *        - if miss → stream from mmap / disk
 *        - compute
 *        - optionally keep in cache (LRU)
 */

/* Placeholder — real implementation coming later */
void rabbit_moe_forward_stub(RabbitModel *model) {
    (void)model;
    printf("[moe] expert streaming not implemented yet.\n");
}
