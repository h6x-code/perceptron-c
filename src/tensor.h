#pragma once
#include <stddef.h>

typedef struct { int rows, cols; float *data; } Tensor;

Tensor tensor_alloc(int rows, int cols);

void   tensor_free(Tensor *t);
void   tensor_zero(Tensor *t);
void   tensor_randu(Tensor *t, unsigned seed);
