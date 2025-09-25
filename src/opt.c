#include "opt.h"
#include <stdlib.h>

int sgd_momentum_init(SGD_Momentum *opt, int L, Tensor *W, Tensor *b, float mu) {
    opt->mu = mu;
    opt->L  = L;
    opt->vW = (Tensor*)malloc((size_t)L * sizeof(Tensor));
    opt->vb = (Tensor*)malloc((size_t)L * sizeof(Tensor));
    if (!opt->vW || !opt->vb) { free(opt->vW); free(opt->vb); return 1; }
    for (int l = 0; l < L; ++l) {
        opt->vW[l] = tensor_alloc(W[l].rows, W[l].cols);
        opt->vb[l] = tensor_alloc(1, b[l].cols);
        tensor_zero(&opt->vW[l]);
        tensor_zero(&opt->vb[l]);
    }
    return 0;
}

void sgd_momentum_free(SGD_Momentum *opt) {
    if (!opt) return;
    if (opt->vW) { for (int l=0;l<opt->L;l++) tensor_free(&opt->vW[l]); free(opt->vW); }
    if (opt->vb) { for (int l=0;l<opt->L;l++) tensor_free(&opt->vb[l]); free(opt->vb); }
    opt->vW = opt->vb = NULL; opt->L = 0; opt->mu = 0.0f;
}

void zero_grads(Tensor *dW, Tensor *db, int L) {
    for (int l = 0; l < L; ++l) { tensor_zero(&dW[l]); tensor_zero(&db[l]); }
}

void sgd_step_params(Tensor *W, Tensor *b, Tensor *dW, Tensor *db, int L, float lr) {
    for (int l = 0; l < L; ++l) {
        int nW = W[l].rows * W[l].cols;
        int nb = b[l].cols;
        for (int i = 0; i < nW; ++i) W[l].data[i] -= lr * dW[l].data[i];
        for (int j = 0; j < nb; ++j) b[l].data[j] -= lr * db[l].data[j];
    }
}

void sgd_momentum_step(SGD_Momentum *opt, Tensor *W, Tensor *b, Tensor *dW, Tensor *db, float lr) {
    float mu = opt->mu;
    for (int l = 0; l < opt->L; ++l) {
        int nW = W[l].rows * W[l].cols;
        int nb = b[l].cols;
        // velocities
        for (int i = 0; i < nW; ++i) {
            opt->vW[l].data[i] = mu * opt->vW[l].data[i] + dW[l].data[i];
            W[l].data[i]      -= lr * opt->vW[l].data[i];
        }
        for (int j = 0; j < nb; ++j) {
            opt->vb[l].data[j] = mu * opt->vb[l].data[j] + db[l].data[j];
            b[l].data[j]      -= lr * opt->vb[l].data[j];
        }
    }
}
