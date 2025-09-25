#include "tensor.h"
#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper: forward pass of 2-2-2 network, returns loss; also optionally dumps intermediates
static float forward_2_2_2(const Tensor *x,
                           const Tensor *W1, const Tensor *b1,
                           const Tensor *W2, const Tensor *b2,
                           int y,
                           Tensor *z1, Tensor *a1, Tensor *logits)
{
    // z1 = x W1 + b1
    dense_forward(x, W1, b1, z1);

    // a1 = ReLU(z1) — allocate its own buffer and copy z1 into it
    *a1 = tensor_alloc(z1->rows, z1->cols);
    for (int i = 0; i < z1->rows * z1->cols; i++) {
        a1->data[i] = z1->data[i];
    }
    relu_inplace(a1);

    // logits = a1 W2 + b2
    dense_forward(a1, W2, b2, logits);

    // loss from logits
    return cross_entropy_from_logits(logits, y);
}


// Analytic backprop through the same network
static void backward_2_2_2(const Tensor *x,
                           const Tensor *W1, const Tensor *b1,
                           const Tensor *W2, const Tensor *b2,
                           const Tensor *z1, const Tensor *a1, const Tensor *logits,
                           int y,
                           Tensor *dW1, Tensor *db1, Tensor *dW2, Tensor *db2)
{
    (void)b1; (void)b2; (void)z1; // currently unused

    // dlogits = softmax(logits) - onehot(y)
    Tensor dlogits = tensor_alloc(1, 2);
    softmax_ce_backward_from_logits(logits, y, &dlogits);

    // Layer 2 backward
    Tensor da1 = tensor_alloc(1, 2);
    dense_backward(a1, W2, &dlogits, &da1, dW2, db2);

    // ReLU backward: mask using pre-activation z1 (more stable than using a1)
    relu_backward_inplace(z1, &da1);


    // Layer 1 backward
    Tensor dx = tensor_alloc(1, 2);
    dense_backward(x, W1, &da1, &dx, dW1, db1);

    tensor_free(&dx);
    tensor_free(&da1);
    tensor_free(&dlogits);
}


// Finite difference gradient for a single parameter array
static void finite_diff_param(float *param, int count, float eps,
                              float (*loss_fn)(void *ctx),
                              void *ctx,
                              float *out_grad)
{
    for (int i = 0; i < count; i++) {
        float orig = param[i];

        param[i] = orig + eps;
        float Lp = loss_fn(ctx);

        param[i] = orig - eps;
        float Lm = loss_fn(ctx);

        out_grad[i] = (Lp - Lm) / (2.0f * eps);
        param[i] = orig; // restore
    }
}

// Context for the loss function wrapper
typedef struct {
    Tensor x, W1, b1, W2, b2;
    int y;
} GCtx;

static float loss_wrapper(void *vctx) {
    GCtx *c = (GCtx*)vctx;

    Tensor z1  = tensor_alloc(1, 2);
    Tensor a1  = (Tensor){0, 0, NULL};
    Tensor log = tensor_alloc(1, 2);

    float L = forward_2_2_2(&c->x, &c->W1, &c->b1, &c->W2, &c->b2, c->y, &z1, &a1, &log);

    // cleanup
    tensor_free(&log);
    tensor_free(&a1);
    tensor_free(&z1);

    return L;
}

static float rel_error(float a, float b) {
    float denom = fmaxf(1e-6f, fabsf(a) + fabsf(b));
    return fabsf(a - b) / denom;
}

int gradcheck_run(void) {
    // Fixed input and labels
    Tensor x  = tensor_alloc(1, 2);
    x.data[0] = 0.7f; x.data[1] = -1.2f;
    int y = 1;

    // Parameters (init deterministic)
    Tensor W1 = tensor_alloc(2, 2);
    Tensor b1 = tensor_alloc(1, 2);
    Tensor W2 = tensor_alloc(2, 2);
    Tensor b2 = tensor_alloc(1, 2);

    // Set some values (small and varied)
    float W1_init[] = { 0.10f, -0.20f,
                        0.05f,  0.30f };
    float b1_init[] = { 0.25f, -0.35f };
    float W2_init[] = { -0.40f, 0.25f,
                         0.15f, 0.05f };
    float b2_init[] = { 0.02f, -0.01f };
    memcpy(W1.data, W1_init, sizeof(W1_init));
    memcpy(b1.data, b1_init, sizeof(b1_init));
    memcpy(W2.data, W2_init, sizeof(W2_init));
    memcpy(b2.data, b2_init, sizeof(b2_init));

    // Forward to capture intermediates for analytic path
    Tensor z1  = tensor_alloc(1, 2);
    Tensor a1  = (Tensor){0, 0, NULL};
    Tensor log = tensor_alloc(1, 2);
    float L = forward_2_2_2(&x, &W1, &b1, &W2, &b2, y, &z1, &a1, &log);

    // Analytic gradients
    Tensor dW1 = tensor_alloc(2, 2);
    Tensor db1 = tensor_alloc(1, 2);
    Tensor dW2 = tensor_alloc(2, 2);
    Tensor db2 = tensor_alloc(1, 2);
    backward_2_2_2(&x, &W1, &b1, &W2, &b2, &z1, &a1, &log, y, &dW1, &db1, &dW2, &db2);

    // Numeric gradients
    GCtx ctx = { x, W1, b1, W2, b2, y };
    const float eps = 5e-4f;

    float *gW1n = (float*)malloc(4 * sizeof(float));
    float *gb1n = (float*)malloc(2 * sizeof(float));
    float *gW2n = (float*)malloc(4 * sizeof(float));
    float *gb2n = (float*)malloc(2 * sizeof(float));

    finite_diff_param(W1.data, 4, eps, loss_wrapper, &ctx, gW1n);
    finite_diff_param(b1.data, 2, eps, loss_wrapper, &ctx, gb1n);
    finite_diff_param(W2.data, 4, eps, loss_wrapper, &ctx, gW2n);
    finite_diff_param(b2.data, 2, eps, loss_wrapper, &ctx, gb2n);

    // Compare and report max relative error
    float max_err = 0.0f;

    for (int i = 0; i < 4; i++) {
        float e = rel_error(dW1.data[i], gW1n[i]);
        if (e > max_err) max_err = e;
    }
    for (int i = 0; i < 2; i++) {
        float e = rel_error(db1.data[i], gb1n[i]);
        if (e > max_err) max_err = e;
    }
    for (int i = 0; i < 4; i++) {
        float e = rel_error(dW2.data[i], gW2n[i]);
        if (e > max_err) max_err = e;
    }
    for (int i = 0; i < 2; i++) {
        float e = rel_error(db2.data[i], gb2n[i]);
        if (e > max_err) max_err = e;
    }

    printf("[gradcheck] loss=%.6f max_rel_error=%.3e\n", L, max_err);

    // Cleanup
    free(gb2n); free(gW2n); free(gb1n); free(gW1n);
    tensor_free(&dW2); tensor_free(&db2);
    tensor_free(&dW1); tensor_free(&db1);
    tensor_free(&log); tensor_free(&a1); tensor_free(&z1);
    tensor_free(&b2); tensor_free(&W2);
    tensor_free(&b1); tensor_free(&W1);
    tensor_free(&x);

    // Acceptance threshold
    return (max_err < 1e-4f) ? 0 : 1;
}
