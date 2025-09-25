#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data.h"
#include "nn.h"
#include "opt.h"
#include "tensor.h"

static void usage(void) {
    puts("perceptron - C11 MLP\n"
         "Usage:\n"
         "  ./perceptron --help\n"
         "  ./perceptron train   [flags]\n"
         "  ./perceptron predict [flags]\n"
         "\nDefaults: --layers 1 --units 128 --epochs 10 --lr 0.01 --batch 32 --seed 1337");
}

static void clip_inplace(float *g, int n, float limit) {
    for (int i = 0; i < n; ++i) {
        if (g[i] >  limit) g[i] =  limit;
        if (g[i] < -limit) g[i] = -limit;
    }
}

// Scale tensor data in-place from [-1,1] to [-a, a]
static void tensor_scale_range(Tensor *t, float a) {
    int n = t->rows * t->cols;
    for (int i = 0; i < n; ++i) t->data[i] *= a;
}

// He-uniform initializer for a (fan_in x fan_out) weight matrix
static void he_uniform_init(Tensor *W, int fan_in, unsigned seed) {
    // Start from uniform in [-1,1], then scale to [-sqrt(6/fan_in), +sqrt(6/fan_in)]
    tensor_randu(W, seed);
    float a = sqrtf(6.0f / (float)fan_in);
    tensor_scale_range(W, a);
}

// tiny deterministic RNG for shuffling (LCG)
static inline uint32_t lcg32(uint32_t *s){ *s = (*s * 1664525u) + 1013904223u; return *s; }

// Fisher–Yates shuffle of indices[0..n-1], deterministic by seed
static void shuffle_indices(int *idx, int n, unsigned seed) {
    uint32_t s = seed;
    for (int i = n - 1; i > 0; --i) {
        uint32_t r = lcg32(&s);
        int j = (int)(r % (uint32_t)(i + 1));
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
}

// Argmax over a 1xK logits/probs vector
static int argmax(const Tensor *t) {
    int n = t->rows * t->cols;
    int best = 0;
    float v = t->data[0];
    for (int i = 1; i < n; i++) {
        if (t->data[i] > v) { v = t->data[i]; best = i; }
    }
    return best;
}

// Train a fixed 2–2–2 MLP on XOR with SGD
static void train_xor(int epochs, float lr, unsigned seed) {
    // Load XOR (n=4, d=2, k=2)
    Dataset d = load_dataset("xor");
    if (d.n == 0) { fprintf(stderr, "dataset load failed\n"); return; }

    // Parameters
    Tensor W1 = tensor_alloc(2, 2);
    Tensor b1 = tensor_alloc(1, 2);
    Tensor W2 = tensor_alloc(2, 2);
    Tensor b2 = tensor_alloc(1, 2);

    // New: He-uniform for ReLU layers
    he_uniform_init(&W1, /*fan_in=*/2, seed);
    he_uniform_init(&W2, /*fan_in=*/2, seed + 1);

    // New: small positive biases for hidden ReLU to avoid dead units
    b1.data[0] = 0.10f;
    b1.data[1] = 0.10f;

    // Output biases can be zero
    b2.data[0] = 0.0f;
    b2.data[1] = 0.0f;

    // Grad buffers
    Tensor dW1 = tensor_alloc(2, 2);
    Tensor db1 = tensor_alloc(1, 2);
    Tensor dW2 = tensor_alloc(2, 2);
    Tensor db2 = tensor_alloc(1, 2);

    // Work buffers (forward)
    Tensor x  = tensor_alloc(1, 2);
    Tensor z1 = tensor_alloc(1, 2);
    Tensor a1 = tensor_alloc(1, 2);
    Tensor z2 = tensor_alloc(1, 2);

    // Work buffers (backward)
    Tensor dz2 = tensor_alloc(1, 2);
    Tensor da1 = tensor_alloc(1, 2);
    Tensor dx  = tensor_alloc(1, 2);

    int idx[4] = {0,1,2,3};

    for (int e = 1; e <= epochs; ++e) {
        shuffle_indices(idx, d.n, seed + (unsigned)e);

        float loss_sum = 0.0f;
        int correct = 0;

        for (int t = 0; t < d.n; ++t) {
            int i = idx[t];

            // Load sample into x (1x2)
            x.data[0] = d.X[2*i + 0];
            x.data[1] = d.X[2*i + 1];
            int y = d.y[i];

            // Forward: z1 -> a1 -> z2
            dense_forward(&x, &W1, &b1, &z1);

            // a1 = LeakyReLU(z1)
            for (int k = 0; k < 2; ++k) a1.data[k] = z1.data[k];
            leaky_relu_inplace(&a1, 0.1f);
;
            dense_forward(&a1, &W2, &b2, &z2);

            // Loss and accuracy
            float L = cross_entropy_from_logits(&z2, y);
            loss_sum += L;

            if (isnan(L) || isinf(L)) {
                fprintf(stderr, "[train] numerical issue (loss=%f). Try smaller --lr.\n", L);
                break;
            }

            // argmax over logits (same result as softmax argmax, no mutation)
            if (argmax(&z2) == y) {
                correct++;
            }

            // Backward: logits -> a1
            softmax_ce_backward_from_logits(&z2, y, &dz2);
            dense_backward(&a1, &W2, &dz2, &da1, &dW2, &db2);

            // Backward: LeakyReLU (use pre-activation z1)
            leaky_relu_backward_inplace(&z1, &da1, 0.1f);
            dense_backward(&x, &W1, &da1, &dx, &dW1, &db1);

            // Clip grads to avoid blow-ups on rare outliers
            clip_inplace(dW2.data, 4, 5.0f);
            clip_inplace(db2.data, 2, 5.0f);
            clip_inplace(dW1.data, 4, 5.0f);
            clip_inplace(db1.data, 2, 5.0f);

            // SGD step
            sgd_step(W2.data, dW2.data, 4, lr);
            sgd_step(b2.data, db2.data, 2, lr);
            sgd_step(W1.data, dW1.data, 4, lr);
            sgd_step(b1.data, db1.data, 2, lr);
        }

        float acc = (float)correct / (float)d.n;
        printf("[epoch %3d] loss=%.6f acc=%.2f%%\n", e, loss_sum / (float)d.n, acc * 100.0f);

        // early exit if solid accuracy
        if (acc >= 0.95f) {
            printf("[train] reached >=95%% accuracy — stopping early.\n");
            break;
        }
    }

    // cleanup
    tensor_free(&dx);  tensor_free(&da1); tensor_free(&dz2);
    tensor_free(&z2);  tensor_free(&a1);  tensor_free(&z1); tensor_free(&x);
    tensor_free(&dW2); tensor_free(&db2); tensor_free(&dW1); tensor_free(&db1);
    tensor_free(&b2);  tensor_free(&W2);  tensor_free(&b1);  tensor_free(&W1);
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0) {
        usage();
        return 0;
    }

    if (strcmp(argv[1], "train") == 0) {
        // Minimal flags for now: --epochs N --lr X --seed S --dataset xor
        int    epochs = 2000;
        float  lr     = 0.1f;
        unsigned seed = 1337;
        const char *dataset = "xor";

        // simple flag parse (very minimal)
        for (int a = 2; a < argc; ++a) {
            if (!strcmp(argv[a], "--epochs") && a+1 < argc) { epochs = atoi(argv[++a]); }
            else if (!strcmp(argv[a], "--lr") && a+1 < argc) { lr = (float)atof(argv[++a]); }
            else if (!strcmp(argv[a], "--seed") && a+1 < argc) { seed = (unsigned)strtoul(argv[++a], NULL, 10); }
            else if (!strcmp(argv[a], "--dataset") && a+1 < argc) { dataset = argv[++a]; }
        }

        if (strcmp(dataset, "xor") != 0) {
            fprintf(stderr, "Only --dataset xor is supported at this step.\n");
            return 1;
        }

        printf("[train] dataset=%s epochs=%d lr=%.4f seed=%u\n", dataset, epochs, lr, seed);
        train_xor(epochs, lr, seed);
        return 0;
    }


    if (strcmp(argv[1], "predict") == 0) {
        puts("[predict] subcommand recognized (flags parsed later).");
        return 0;
    }

    if (strcmp(argv[1], "tensor-test") == 0) {
        if (argc < 3) { fprintf(stderr, "usage: ./perceptron tensor-test <seed>\n"); return 2; }
        unsigned seed = (unsigned)strtoul(argv[2], NULL, 10);
        Tensor t = tensor_alloc(2, 3);
        tensor_zero(&t); float s0=0; for(int i=0;i<6;i++) s0 += t.data[i];
        tensor_randu(&t, seed); float s1=0; for(int i=0;i<6;i++) s1 += t.data[i];
        printf("[tensor] seed=%u sum_zero=%.1f sum_rand=%.6f\n", seed, s0, s1);
        tensor_free(&t); return 0;
    }

    if (strcmp(argv[1], "nn-test") == 0) {
        nn_test(); 
        return 0; 
    }

    if (strcmp(argv[1], "loss-test") == 0) {
        // Build a 1x2 logits vector
        Tensor logits = tensor_alloc(1, 2);
        logits.data[0] = -0.2f;  // class 0
        logits.data[1] =  0.1f;  // class 1
        int y = 1;               // true label is class 1

        float before = cross_entropy_from_logits(&logits, y);

        // Nudge the true class logit upward by +0.5
        logits.data[y] += 0.5f;

        float after = cross_entropy_from_logits(&logits, y);

        printf("[loss] before=%.6f after=%.6f (expect after < before)\n", before, after);

        tensor_free(&logits);
        return 0;
    }

    if (strcmp(argv[1], "gradcheck") == 0) {
        int rc = gradcheck_run();
        if (rc == 0) {
            puts("[gradcheck] OK (max_rel_error < 1e-4)");
        } else {
            puts("[gradcheck] FAIL (max_rel_error >= 1e-4)");
        }
        return rc;
    }

    fprintf(stderr, "error: unknown subcommand '%s'\n", argv[1]);
    usage();
    return 1;
}
