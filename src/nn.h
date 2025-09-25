#pragma once
#include "tensor.h"

void dense_forward(const Tensor *x, const Tensor *W, const Tensor *b, Tensor *out);
void relu_inplace(Tensor *t);
void softmax_inplace(Tensor *t);
void nn_test(void);  // tiny demo
