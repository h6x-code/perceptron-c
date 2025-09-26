#ifndef OPT_H
#define OPT_H

#include "tensor.h"

// Simple SGD + momentum optimizer state.
// mu = momentum coefficient in [0, 1)
typedef struct {
    int L;          // number of layers
    Tensor *vW;     // velocity for weights, length L
    Tensor *vB;     // velocity for biases,  length L
    float mu;       // momentum coefficient
} SGD_Momentum;

// Initialize momentum buffers shaped like W[l]/b[l] and zero them.
// Returns 0 on success, nonzero on OOM.
int sgd_momentum_init(SGD_Momentum *opt, int L,
                      const Tensor *W, const Tensor *b,
                      float mu);

// One SGD+momentum update:
// v = mu * v - lr * grad
// param += v
void sgd_momentum_step(SGD_Momentum *opt,
                       Tensor *W, Tensor *B,
                       Tensor *dW, Tensor *dB,
                       float lr);

// Free all momentum buffers.
void sgd_momentum_free(SGD_Momentum *opt);

#endif // OPT_H
