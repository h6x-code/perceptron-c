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

    m->dims = (int*)malloc((m->L + 1) * sizeof(int));
    if (!m->dims) return -1;
    m->dims[0] = d_in;
    for (int i = 0; i < n_hidden; ++i) m->dims[i+1] = hidden[i];
    m->dims[m->L] = d_out;

    m->W = (Tensor*)malloc(m->L * sizeof(Tensor));
    m->b = (Tensor*)malloc(m->L * sizeof(Tensor));
    m->z = (Tensor*)malloc(m->L * sizeof(Tensor));
    m->a = (Tensor*)malloc((m->L + 1) * sizeof(Tensor));
    if (!m->W || !m->b || !m->z || !m->a) return -2;

    // Initialize structs to a known zero state to keep tensor_free safe
    for (int l = 0; l < m->L; ++l) {
        m->W[l] = (Tensor){0,0,NULL};
        m->b[l] = (Tensor){0,0,NULL};
        m->z[l] = (Tensor){0,0,NULL};
    }
    for (int l = 0; l <= m->L; ++l) {
        m->a[l] = (Tensor){0,0,NULL};
    }

    // a[0] is a 1 x d_in copy of the input
    m->a[0] = tensor_alloc(1, d_in);

    // Allocate per-layer params and buffers
    for (int l = 0; l < m->L; ++l) {
        int din  = m->dims[l];
        int dout = m->dims[l+1];

        m->W[l] = tensor_alloc(din, dout);
        m->b[l] = tensor_alloc(1,  dout);
        he_uniform_init(&m->W[l], din, seed + (unsigned)l);

        // small positive bias for hidden; zero for output
        for (int j = 0; j < dout; ++j) {
            m->b[l].data[j] = (l < m->L - 1) ? 0.10f : 0.0f;
        }

        m->z[l]     = tensor_alloc(1, dout);      // pre-activation z[l]
        m->a[l + 1] = tensor_alloc(1, dout);      // activation a[l+1]
    }
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

    for (int l = 0; l < m->L; ++l) {
        // z[l] = a[l] W[l] + b[l]
        dense_forward(&m->a[l], &m->W[l], &m->b[l], &m->z[l]);

        if (l == m->L - 1) {
            // last layer: logits -> out
            for (int j = 0; j < m->d_out; ++j) out->data[j] = m->z[l].data[j];
        } else {
            // hidden: a[l+1] = LeakyReLU(z[l])
            for (int j = 0; j < m->dims[l+1]; ++j) m->a[l+1].data[j] = m->z[l].data[j];
            leaky_relu_inplace(&m->a[l+1], 0.1f);
        }
    }
}


void mlp_backward_from_logits(MLP *m, const Tensor *x, int y,
                              Tensor *dW, Tensor *db, float leaky_alpha)
{
    (void)x; // input is cached in a[0]

    Tensor dcur = tensor_alloc(1, m->d_out);
    softmax_ce_backward_from_logits(&m->z[m->L - 1], y, &dcur);

    for (int l = m->L - 1; l >= 0; --l) {
        Tensor dx = tensor_alloc(1, m->dims[l]);
        dense_backward(&m->a[l], &m->W[l], &dcur, &dx, &dW[l], &db[l]);

        if (l > 0) { // apply activation derivative to da[l] using pre-activation z[l-1]
            leaky_relu_backward_inplace(&m->z[l - 1], &dx, leaky_alpha);
        }

        tensor_free(&dcur);
        dcur = dx;
    }
    tensor_free(&dcur);
}

void mlp_sgd_step(MLP *m, Tensor *dW, Tensor *db, float lr) {
    for (int l = 0; l < m->L; ++l) {
        int nW = m->W[l].rows * m->W[l].cols;
        int nb = m->b[l].cols;
        for (int i = 0; i < nW; ++i) m->W[l].data[i] -= lr * dW[l].data[i];
        for (int j = 0; j < nb; ++j) m->b[l].data[j] -= lr * db[l].data[j];
    }
}
