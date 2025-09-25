#pragma once
#include "tensor.h"

// Minimal multi-layer perceptron (MLP) with LeakyReLU hidden layers.
typedef struct {
    int L;        // number of layers (including output)
    int d_in;     // input dim
    int d_out;    // output dim
    int *dims;    // dims[0]=d_in, dims[1]=h1, ..., dims[L-1]=h_{L-1}, dims[L]=d_out  (size L+1)

    Tensor *W;    // W[l] has shape (dims[l] x dims[l+1])
    Tensor *b;    // b[l] has shape (1 x dims[l+1])

    // Work buffers for forward/backward (per layer)
    Tensor *z;    // pre-activations (1 x dims[l+1])
    Tensor *a;    // activations   (1 x dims[l+1]), with a[0] used as 1xd_in copy of input
} MLP;

int  mlp_init(MLP *m, int d_in, int d_out, const int *hidden, int n_hidden, unsigned seed);
void mlp_free(MLP *m);

// Forward: compute logits for a single sample x(1 x d_in) into out(1 x d_out)
void mlp_forward_logits(MLP *m, const Tensor *x, Tensor *out);

// Backward from logits + label, producing per-layer grads (provided as arrays)
void mlp_backward_from_logits(MLP *m, const Tensor *x, int y,
                              Tensor *dW, Tensor *db, float leaky_alpha);

// Utility: He-uniform init for a given W (fan_in = dims[l])
void he_uniform_init(Tensor *W, int fan_in, unsigned seed);

// Simple SGD step over all layers
void mlp_sgd_step(MLP *m, Tensor *dW, Tensor *db, float lr);
