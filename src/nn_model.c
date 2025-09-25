#include "nn_model.h"
#include "nn.h"       // dense_forward, leaky_relu_inplace/backward, softmax_ce_backward_from_logits
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void tensor_scale_range(Tensor *t, float a) {
    int n = t->rows * t->cols;
    for (int i = 0; i < n; ++i) t->data[i] *= a;
}

void he_uniform_init(Tensor *W, int fan_in, unsigned seed) {
    tensor_randu(W, seed);                   // [-1,1]
    float a = sqrtf(6.0f / (float)fan_in);   // He-uniform half-range
    tensor_scale_range(W, a);                // [-a, a]
}

int mlp_init(MLP *m, int d_in, int d_out, const int *hidden, int n_hidden, unsigned seed) {
    m->L     = n_hidden + 1;
    m->d_in  = d_in;
    m->d_out = d_out;

    // dims array
    m->dims = (int*)malloc((m->L + 1) * sizeof(int));
    if (!m->dims) return -1;
    m->dims[0] = d_in;
    for (int i = 0; i < n_hidden; ++i) m->dims[i+1] = hidden[i];
    m->dims[m->L] = d_out;

    // allocate parameter arrays
    m->W = (Tensor*)malloc(m->L * sizeof(Tensor));
    m->b = (Tensor*)malloc(m->L * sizeof(Tensor));
    m->z = (Tensor*)malloc(m->L * sizeof(Tensor));
    m->a = (Tensor*)malloc((m->L + 1) * sizeof(Tensor)); // a[0] holds input copy
    if (!m->W || !m->b || !m->z || !m->a) return -2;

    for (int l = 0; l < m->L; ++l) {
        int din = m->dims[l];
        int dout = m->dims[l+1];
        m->W[l] = tensor_alloc(din, dout);
        m->b[l] = tensor_alloc(1, dout);
        he_uniform_init(&m->W[l], din, seed + (unsigned)l);
        // small positive bias for hidden; zero for output
        for (int j = 0; j < dout; ++j) {
            m->b[l].data[j] = (l < m->L - 1) ? 0.10f : 0.0f;
        }
        m->z[l] = tensor_alloc(1, dout);
        m->a[l] = tensor_alloc(1, dout);  // will hold activations per layer
    }
    // a[0] is special: copy of input (1 x d_in)
    m->a[0] = tensor_alloc(1, d_in);
    return 0;
}

void mlp_free(MLP *m) {
    if (!m) return;
    if (m->W) { for (int l=0;l<m->L;l++) tensor_free(&m->W[l]); free(m->W); }
    if (m->b) { for (int l=0;l<m->L;l++) tensor_free(&m->b[l]); free(m->b); }
    if (m->z) { for (int l=0;l<m->L;l++) tensor_free(&m->z[l]); free(m->z); }
    if (m->a) { for (int l=0;l<=m->L;l++) tensor_free(&m->a[l]); free(m->a); }
    if (m->dims) { free(m->dims); }
    memset(m, 0, sizeof(*m));
}

void mlp_forward_logits(MLP *m, const Tensor *x, Tensor *out) {
    // a[0] = x
    for (int i = 0; i < m->d_in; ++i) m->a[0].data[i] = x->data[i];

    // layers 1..L
    for (int l = 0; l < m->L; ++l) {
        dense_forward(&m->a[l], &m->W[l], &m->b[l], &m->z[l]);

        // last layer: logits (no activation)
        if (l == m->L - 1) {
            // copy logits to out
            for (int j = 0; j < m->d_out; ++j) out->data[j] = m->z[l].data[j];
        } else {
            // hidden activation = LeakyReLU(z)
            for (int j = 0; j < m->dims[l+1]; ++j) m->a[l+1].data[j] = m->z[l].data[j];
            leaky_relu_inplace(&m->a[l+1], 0.1f);
        }
    }
}

void mlp_backward_from_logits(MLP *m, const Tensor *x, int y,
                              Tensor *dW, Tensor *db, float leaky_alpha)
{
    (void)x;    // input is already in m->a[0] from the forward pass
    // Upstream gradient at logits
    Tensor dcur = tensor_alloc(1, m->d_out);
    softmax_ce_backward_from_logits(&m->z[m->L - 1], y, &dcur);

    // Backward through layers L..1
    for (int l = m->L - 1; l >= 0; --l) {
        // Dense backward for layer l: a[l] -> z[l]
        Tensor dx = tensor_alloc(1, m->dims[l]);
        dense_backward(&m->a[l], &m->W[l], &dcur, &dx, &dW[l], &db[l]);

        // If not the first layer, apply LeakyReLU backward to dx (which is da[l])
        if (l > 0) {
            leaky_relu_backward_inplace(&m->z[l-1], &dx, leaky_alpha);
        }

        // Next upstream gradient is dx
        tensor_free(&dcur);
        dcur = dx;
    }

    tensor_free(&dcur); // after l==0
}

void mlp_sgd_step(MLP *m, Tensor *dW, Tensor *db, float lr) {
    for (int l = 0; l < m->L; ++l) {
        int nW = m->W[l].rows * m->W[l].cols;
        int nb = m->b[l].cols;
        for (int i = 0; i < nW; ++i) m->W[l].data[i] -= lr * dW[l].data[i];
        for (int j = 0; j < nb; ++j) m->b[l].data[j] -= lr * db[l].data[j];
    }
}
