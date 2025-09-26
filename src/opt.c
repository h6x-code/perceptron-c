#include <stdlib.h>
#include <string.h>
#include "opt.h"

int sgd_momentum_init(SGD_Momentum *opt, int L,
                      const Tensor *W, const Tensor *b,
                      float mu)
{
    if (!opt || L <= 0 || !W || !b) return -1;
    memset(opt, 0, sizeof(*opt));
    opt->L  = L;
    opt->mu = mu;

    opt->vW = (Tensor*)calloc((size_t)L, sizeof(Tensor));
    opt->vB = (Tensor*)calloc((size_t)L, sizeof(Tensor));
    if (!opt->vW || !opt->vB) return -1;

    for (int l = 0; l < L; ++l) {
        // Weight velocity matches W[l] (rows x cols)
        opt->vW[l] = tensor_alloc(W[l].rows, W[l].cols);
        // Bias velocity matches b[l] (1 x cols)
        opt->vB[l] = tensor_alloc(b[l].rows, b[l].cols);
        if (!opt->vW[l].data || !opt->vB[l].data) return -1;

        tensor_zero(&opt->vW[l]);
        tensor_zero(&opt->vB[l]);
    }
    return 0;
}

void sgd_momentum_step(SGD_Momentum *opt,
                       Tensor *W, Tensor *B,
                       Tensor *dW, Tensor *dB,
                       float lr)
{
    // Assumes arrays W/B/dW/dB are length L with matching shapes.
    if (!opt || !W || !B || !dW || !dB) return;

    const int L = opt->L;
    const float mu = opt->mu;

    for (int l = 0; l < L; ++l) {
        // vW = mu*vW - lr*dW; W += vW
        int nW = W[l].rows * W[l].cols;
        float *vW = opt->vW[l].data;
        float *w  = W[l].data;
        float *gw = dW[l].data;
        for (int i = 0; i < nW; ++i) {
            vW[i] = mu * vW[i] - lr * gw[i];
            w[i] += vW[i];
        }

        // vB = mu*vB - lr*dB; B += vB
        int nB = B[l].rows * B[l].cols;
        float *vB = opt->vB[l].data;
        float *b  = B[l].data;
        float *gb = dB[l].data;
        for (int i = 0; i < nB; ++i) {
            vB[i] = mu * vB[i] - lr * gb[i];
            b[i] += vB[i];
        }
    }
}

void sgd_momentum_free(SGD_Momentum *opt)
{
    if (!opt) return;
    if (opt->vW) {
        for (int l = 0; l < opt->L; ++l) tensor_free(&opt->vW[l]);
        free(opt->vW);
        opt->vW = NULL;
    }
    if (opt->vB) {
        for (int l = 0; l < opt->L; ++l) tensor_free(&opt->vB[l]);
        free(opt->vB);
        opt->vB = NULL;
    }
    opt->L = 0;
}
