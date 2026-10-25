#include "rabbit.h"
#include <stdio.h>

/*
 * Quantization kernels.
 * Target: int4 and more aggressive formats so the dense part
 * can fit in the tiny RAM budget of a 4 GB machine.
 */

void rabbit_quant_stub(void) {
    printf("[quant] quantization kernels not implemented yet.\n");
}
