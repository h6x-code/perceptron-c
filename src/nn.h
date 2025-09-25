#pragma once
#include "tensor.h"

// Existing forward APIs
void dense_forward(const Tensor *x, const Tensor *W, const Tensor *b, Tensor *out);
void relu_inplace(Tensor *t);
void softmax_inplace(Tensor *t);

// loss helpers (logits-based)
float logsumexp(const Tensor *logits);
float cross_entropy_from_logits(const Tensor *logits, int true_label);

// Backward primitives
// Given logits (1xK) and label, produce gradient wrt logits: dlogits = softmax(logits) - onehot(label)
void softmax_ce_backward_from_logits(const Tensor *logits, int true_label, Tensor *dlogits);

// In-place ReLU backward: zero grad where pre-activation <= 0
// 'activation' is expected to be the PRE-activation (z), not the post-ReLU output.
void relu_backward_inplace(const Tensor *activation, Tensor *dactivation);

// Dense backward: x(1xd), W(dxm), dout(1xm) -> dx(1xd), dW(dxm), db(1xm)
void dense_backward(const Tensor *x, const Tensor *W, const Tensor *dout,
                    Tensor *dx, Tensor *dW, Tensor *db);

// Tiny demos/tests
void nn_test(void);

// Gradcheck runner
int gradcheck_run(void);
