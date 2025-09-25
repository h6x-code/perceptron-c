#include "nn.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// Forward pass for a dense layer: out = xW + b
void dense_forward(const Tensor *x, const Tensor *W, const Tensor *b, Tensor *o) {
    assert(W && b && x && o);
    assert(W->rows > 0 && W->cols > 0);
    assert(b->rows == 1 && b->cols == W->cols);
    assert(x->rows == 1 && x->cols == W->rows);
    assert(o->rows == 1 && o->cols == W->cols);

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

void leaky_relu_inplace(Tensor *t, float alpha) {
    int n = t->rows * t->cols;
    for (int i = 0; i < n; i++) {
        float v = t->data[i];
        t->data[i] = (v > 0.0f) ? v : alpha * v;
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

// Compute log-sum-exp over a 1xK logits tensor.
float logsumexp(const Tensor *logits) {
    int n = logits->rows * logits->cols;      // expect 1xK
    const float *z = logits->data;

    // max in float is fine
    float m = z[0];
    for (int i = 1; i < n; i++) {
        if (z[i] > m) m = z[i];
    }

    // accumulate in double for stability
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        acc += exp((double)z[i] - (double)m); // double exp
    }

    return m + (float)log(acc);
}

// Cross-entropy from raw logits and an integer label (0..K-1).
float cross_entropy_from_logits(const Tensor *logits, int true_label) {
    int n = logits->rows * logits->cols;  // expect 1xK
    if (true_label < 0 || true_label >= n) {
        // In a full system we’d handle this more gracefully; keep simple here.
        return NAN;
    }
    const float *z = logits->data;
    float lse = logsumexp(logits);
    return -z[true_label] + lse;
}

void softmax_ce_backward_from_logits(const Tensor *logits, int true_label, Tensor *dlogits) {
    int n = logits->rows * logits->cols;   // expect 1xK
    const float *z = logits->data;
    float *g = dlogits->data;

    // subtract max (float is fine)
    float m = z[0];
    for (int i = 1; i < n; i++) {
        if (z[i] > m) m = z[i];
    }

    // exponentiate and sum in double
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        g[i] = (float)exp((double)z[i] - (double)m); // store as float
        acc += (double)g[i];
    }
    float inv = 1.0f / (float)acc;
    for (int i = 0; i < n; i++) g[i] *= inv;

    if (true_label >= 0 && true_label < n) {
        g[true_label] -= 1.0f;
    }
}

void relu_backward_inplace(const Tensor *activation, Tensor *dactivation) {
    // Zero gradient where activation == 0 (inactive neurons)
    int n = activation->rows * activation->cols;
    const float *a = activation->data;
    float *da = dactivation->data;

    for (int i = 0; i < n; i++) {
        if (a[i] <= 0.0f) {
            da[i] = 0.0f;
        }
    }
}

void leaky_relu_backward_inplace(const Tensor *preact, Tensor *dact, float alpha) {
    int n = preact->rows * preact->cols;
    const float *z = preact->data;
    float *g = dact->data;
    for (int i = 0; i < n; i++) {
        g[i] *= (z[i] > 0.0f) ? 1.0f : alpha;
    }
}

void dense_backward(const Tensor *x, const Tensor *W, const Tensor *dout,
                    Tensor *dx, Tensor *dW, Tensor *db) {

    assert(x && W && dout && dx && dW && db);
    assert(W->rows > 0 && W->cols > 0);
    assert(x->rows == 1 && x->cols == W->rows);
    assert(dout->rows == 1 && dout->cols == W->cols);
    assert(dx->rows == 1 && dx->cols == W->rows);
    assert(dW->rows == W->rows && dW->cols == W->cols);
    assert(db->rows == 1 && db->cols == W->cols);

    // shapes: x(1xd), W(dxm), dout(1xm)
    int d = W->rows;
    int m = W->cols;

    // db = dout
    for (int j = 0; j < m; j++) {
        db->data[j] = dout->data[j];
    }

    // dW = x^T * dout
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < m; j++) {
            dW->data[i * m + j] = x->data[i] * dout->data[j];
        }
    }

    // dx = dout * W^T
    for (int i = 0; i < d; i++) {
        float s = 0.0f;
        for (int j = 0; j < m; j++) {
            s += dout->data[j] * W->data[i * m + j];
        }
        dx->data[i] = s;
    }
}

void dense_backward_accum(const Tensor *a, const Tensor *W, const Tensor *dout,
                          Tensor *dx, Tensor *dW_acc, Tensor *db_acc)
{
    const int d_in  = W->rows;
    const int d_out = W->cols;

    // dx = dout @ W^T
    for (int i = 0; i < d_in; ++i) {
        float s = 0.0f;
        for (int j = 0; j < d_out; ++j) s += dout->data[j] * W->data[i*d_out + j];
        dx->data[i] = s;
    }

    // dW += a^T @ dout
    for (int i = 0; i < d_in; ++i) {
        const float ai = a->data[i];
        for (int j = 0; j < d_out; ++j) dW_acc->data[i*d_out + j] += ai * dout->data[j];
    }

    // db += dout
    for (int j = 0; j < d_out; ++j) db_acc->data[j] += dout->data[j];
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
