#pragma once
#include "tensor.h"

// Existing forward APIs
void dense_forward(const Tensor *x, const Tensor *W, const Tensor *b, Tensor *out);
void relu_inplace(Tensor *t);
void softmax_inplace(Tensor *t);

// New: loss helpers (logits-based)
float logsumexp(const Tensor *logits);
float cross_entropy_from_logits(const Tensor *logits, int true_label);

// Tiny demos/tests
void nn_test(void);
