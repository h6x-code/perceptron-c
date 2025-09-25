#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data.h"
#include "nn.h"
#include "nn_model.h"
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


// Misc helpers

static int parse_units(const char *s, int *out, int maxn) {
    // returns count parsed; expects "a,b,c"
    int n = 0;
    const char *p = s;
    while (*p && n < maxn) {
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (p == end) break;
        out[n++] = (int)v;
        if (*end == ',') p = end + 1; else { p = end; break; }
    }
    return n;
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


int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0) {
        usage();
        return 0;
    }

    if (strcmp(argv[1], "train") == 0) {
        int    epochs = 2000;
        float  lr     = 0.1f;
        unsigned seed = 1337;
        const char *dataset = "xor";
        int layers_hidden = 1;
        int units_arr[8] = {4};  // default one hidden layer with 4 units
        int units_cnt = 1;

        for (int a = 2; a < argc; ++a) {
            if (!strcmp(argv[a], "--epochs") && a+1 < argc) { epochs = atoi(argv[++a]); }
            else if (!strcmp(argv[a], "--lr") && a+1 < argc) { lr = (float)atof(argv[++a]); }
            else if (!strcmp(argv[a], "--seed") && a+1 < argc) { seed = (unsigned)strtoul(argv[++a], NULL, 10); }
            else if (!strcmp(argv[a], "--dataset") && a+1 < argc) { dataset = argv[++a]; }
            else if (!strcmp(argv[a], "--layers") && a+1 < argc) { layers_hidden = atoi(argv[++a]); }
            else if (!strcmp(argv[a], "--units") && a+1 < argc) { units_cnt = parse_units(argv[++a], units_arr, 8); }
        }

        if (strcmp(dataset, "xor") != 0) {
            fprintf(stderr, "Only --dataset xor is supported at this step.\n");
            return 1;
        }
        if (units_cnt != layers_hidden) {
            fprintf(stderr, "units count (%d) must match --layers (%d)\n", units_cnt, layers_hidden);
            return 1;
        }

        printf("[train] ds=%s epochs=%d lr=%.4f seed=%u layers=%d units=",
            dataset, epochs, lr, seed, layers_hidden);
        for (int i=0;i<units_cnt;i++) printf("%s%d", (i?",":""), units_arr[i]);
        puts("");

        // Train with flexible MLP
        Dataset d = load_dataset("xor");
        MLP m = {0};
        if (mlp_init(&m, /*d_in=*/2, /*d_out=*/2, units_arr, units_cnt, seed) != 0) {
            fprintf(stderr, "mlp_init failed\n"); return 1;
        }

        // Grad buffers aligned to layers
        Tensor *dW = (Tensor*)malloc(m.L * sizeof(Tensor));
        Tensor *db = (Tensor*)malloc(m.L * sizeof(Tensor));
        for (int l=0;l<m.L;l++){ dW[l]=tensor_alloc(m.W[l].rows,m.W[l].cols); db[l]=tensor_alloc(1,m.b[l].cols); }

        Tensor x = tensor_alloc(1, 2);
        Tensor logits = tensor_alloc(1, 2);

        int idx[4] = {0,1,2,3};
        for (int e = 1; e <= epochs; ++e) {
            shuffle_indices(idx, d.n, seed + (unsigned)e);
            int correct = 0;
            float loss_sum = 0.0f;

            for (int t = 0; t < d.n; ++t) {
                int i = idx[t];
                x.data[0] = d.X[2*i+0];
                x.data[1] = d.X[2*i+1];
                int y = d.y[i];

                mlp_forward_logits(&m, &x, &logits);
                loss_sum += cross_entropy_from_logits(&logits, y);
                if (argmax(&logits) == y) correct++;

                mlp_backward_from_logits(&m, &x, y, dW, db, /*alpha=*/0.1f);
                mlp_sgd_step(&m, dW, db, lr);
            }

            float acc = (float)correct / (float)d.n;
            printf("[epoch %3d] loss=%.6f acc=%.2f%%\n", e, loss_sum/(float)d.n, acc*100.0f);
            if (acc >= 0.95f) { puts("[train] reached >=95% accuracy — stopping early."); break; }
        }

        // cleanup
        tensor_free(&logits); tensor_free(&x);
        for (int l=0;l<m.L;l++){ tensor_free(&db[l]); tensor_free(&dW[l]); }
        free(db); free(dW);
        mlp_free(&m);
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
