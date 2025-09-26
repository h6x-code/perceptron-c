#include "tensor.h"

#include <stdint.h>
#include <stdlib.h>

/*
  Tensor allocation:
  - Allocates a contiguous block of floats for rows*cols elements.
  - Returns a value-type Tensor; caller owns the memory and must call tensor_free.
*/
Tensor tensor_alloc(int rows, int cols) {
    Tensor t = { .rows = rows, .cols = cols, .data = NULL };

    size_t n = (size_t)rows * (size_t)cols;
    if (n > 0) {
        t.data = (float *)malloc(n * sizeof(float));
    }
    return t;
}

/*
  Free tensor memory and null the pointer.
  Safe to call multiple times; no-op if data is already NULL.
*/
void tensor_free(Tensor *t)
{
    if (!t) return;

    // if already freed or never allocated, do nothing
    if (t->data) {
        free(t->data);
        t->data = NULL;
    }

    // make future frees safe and signal "empty"
    t->rows = 0;
    t->cols = 0;
}


/*
  Fill tensor with zeros.
*/
void tensor_zero(Tensor *t) {
    size_t n = (size_t)t->rows * (size_t)t->cols;
    for (size_t i = 0; i < n; i++) {
        t->data[i] = 0.0f;
    }
}

/*
  Simple LCG (Linear Congruential Generator) for deterministic pseudo-random numbers.
  Not cryptographic; good enough for initializers and tests.
*/
static inline uint32_t lcg_next(uint32_t *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

/*
  Uniform random initializer in [-1, 1], deterministic with a given seed.
*/
void tensor_randu(Tensor *t, unsigned seed) {
    uint32_t s = seed;
    size_t n = (size_t)t->rows * (size_t)t->cols;

    for (size_t i = 0; i < n; i++) {
        // Use upper 24 bits to make a float in [0,1), then scale/shift to [-1,1].
        float u01 = (float)(lcg_next(&s) >> 8) / 16777216.0f; // 2^24
        t->data[i] = (u01 * 2.0f) - 1.0f;
    }
}
