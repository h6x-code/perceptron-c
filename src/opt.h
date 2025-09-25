#pragma once
#include "tensor.h"

typedef struct {
    float mu;     // momentum coefficient in [0,1)
    Tensor *vW;   // per-layer velocity for W
    Tensor *vb;   // per-layer velocity for b
    int L;        // number of layers
} SGD_Momentum;

// Initialize momentum buffers matching model shapes
int sgd_momentum_init(SGD_Momentum *opt, int L, Tensor *W, Tensor *b, float mu);

// Free momentum buffers
void sgd_momentum_free(SGD_Momentum *opt);

// Vanilla SGD: theta -= lr * grad
void sgd_step_params(Tensor *W, Tensor *b, Tensor *dW, Tensor *db, int L, float lr);

// Momentum SGD: v = mu*v + dtheta; theta -= lr * v
void sgd_momentum_step(SGD_Momentum *opt, Tensor *W, Tensor *b, Tensor *dW, Tensor *db, float lr);

// Zero a stack of gradient tensors
void zero_grads(Tensor *dW, Tensor *db, int L);
