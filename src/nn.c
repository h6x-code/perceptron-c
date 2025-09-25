#include "nn.h"
#include <math.h>

// Forward pass for a dense layer: out = xW + b
void dense_forward(const Tensor *x, const Tensor *W, const Tensor *b, Tensor *o) {
    int d = W->rows;   // input dimension
    int m = W->cols;   // output dimension

    for (int j = 0; j < m; j++) {
        float s = b->data[j];  // start with bias
        for (int i = 0; i < d; i++) {
            s += x->data[i] * W->data[i * m + j];
        }
        o->data[j] = s;
    }
}

// In-place ReLU activation: max(0, x)
void relu_inplace(Tensor *t) {
    int n = t->rows * t->cols;
    for (int i = 0; i < n; i++) {
        if (t->data[i] < 0) {
            t->data[i] = 0;
        }
    }
}

// In-place softmax with numerical stability
void softmax_inplace(Tensor *t) {
    int n = t->rows * t->cols;

    // subtract max to prevent overflow
    float mx = t->data[0];
    for (int i = 1; i < n; i++) {
        if (t->data[i] > mx) {
            mx = t->data[i];
        }
    }

    // exponentiate and accumulate sum
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        t->data[i] = expf(t->data[i] - mx);
        s += t->data[i];
    }

    // normalize
    for (int i = 0; i < n; i++) {
        t->data[i] /= s;
    }
}

// Tiny test function for forward pass
void nn_test(void) {
    Tensor x = tensor_alloc(1, 2);
    Tensor W = tensor_alloc(2, 2);
    Tensor b = tensor_alloc(1, 2);
    Tensor o = tensor_alloc(1, 2);

    // input vector
    x.data[0] = 0.2f;
    x.data[1] = -1.3f;

    // weights (row-major, d x m)
    float w[] = { 1.0f, -0.5f,
                  0.3f,  0.8f };
    for (int i = 0; i < 4; i++) {
        W.data[i] = w[i];
    }

    // biases
    b.data[0] = 0.1f;
    b.data[1] = -0.2f;

    // forward pass
    dense_forward(&x, &W, &b, &o);
    relu_inplace(&o);
    softmax_inplace(&o);

    // check sum of probabilities
    float s = o.data[0] + o.data[1];
    printf("[nn] p=[%.6f, %.6f] sum=%.6f\n", o.data[0], o.data[1], s);

    // cleanup
    tensor_free(&o);
    tensor_free(&b);
    tensor_free(&W);
    tensor_free(&x);
}
