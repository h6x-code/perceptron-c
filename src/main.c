#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "data.h"
#include "io.h"
#include "nn.h"
#include "nn_model.h"
#include "opt.h"
#include "tensor.h"

static void usage(void) {
    puts("perceptron usage:");
    puts("  ./perceptron --help");
    puts("  ./perceptron gradcheck");
    puts("  ./perceptron nn-test | tensor-test <seed>");
    puts("  ./perceptron train --dataset xor [--layers N --units a,b,c]");
    puts("                    [--epochs 500 --lr 0.1 --seed 1337]");
    puts("                    [--val 0.25] [--out path/to/model.bin]");
    puts("  ./perceptron predict --model path/to/model.bin");
}


// Misc helpers
static double now_ms(void) {
#ifdef CLOCK_MONOTONIC
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#else
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
#endif
}

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

static void explain_mnist_rc(int rc) {
    switch (rc) {
        case 1: fprintf(stderr, "MNIST: could not open images or labels file.\n"); break;
        case 2: fprintf(stderr, "MNIST: failed reading image header.\n"); break;
        case 3: fprintf(stderr, "MNIST: failed reading label header.\n"); break;
        case 4: fprintf(stderr, "MNIST: bad magic or count mismatch (need 0x00000803 / 0x00000801 and same N).\n"); break;
        case 5: fprintf(stderr, "MNIST: allocation failed.\n"); break;
        case 6: fprintf(stderr, "MNIST: unexpected EOF while reading images.\n"); break;
        case 7: fprintf(stderr, "MNIST: unexpected EOF while reading labels.\n"); break;
        default: fprintf(stderr, "MNIST: unknown error.\n"); break;
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0) {
        usage();
        return 0;
    }

    if (strcmp(argv[1], "train") == 0) {
        int epochs = 2000;
        float  lr = 0.1f;
        unsigned seed = 1337;
        const char *dataset = "xor";
        int layers_hidden = 1;
        int units_arr[8] = {4};
        int units_cnt = 1;
        float val_frac = 0.0f;
        const char *out_path = NULL;
        int csv_has_header = 0;
        const char *mnist_images = NULL;
        const char *mnist_labels = NULL;
        int limit = 0;
        const char *norm = "minmax"; // default

        for (int a = 2; a < argc; ++a) {
            if (!strcmp(argv[a], "--epochs") && a+1 < argc) { epochs = atoi(argv[++a]); }
            else if (!strcmp(argv[a], "--lr") && a+1 < argc) { lr = (float)atof(argv[++a]); }
            else if (!strcmp(argv[a], "--seed") && a+1 < argc) { seed = (unsigned)strtoul(argv[++a], NULL, 10); }
            else if (!strcmp(argv[a], "--dataset") && a+1 < argc) { dataset = argv[++a]; }
            else if (!strcmp(argv[a], "--layers") && a+1 < argc) { layers_hidden = atoi(argv[++a]); }
            else if (!strcmp(argv[a], "--units") && a+1 < argc) { units_cnt = parse_units(argv[++a], units_arr, 8); }
            else if (!strcmp(argv[a], "--val") && a+1 < argc)    { val_frac = (float)atof(argv[++a]); }
            else if (!strcmp(argv[a], "--out") && a+1 < argc)    { out_path = argv[++a]; }
            else if (!strcmp(argv[a], "--csv-has-header")) { csv_has_header = 1; }
            else if (!strcmp(argv[a], "--mnist-images") && a+1 < argc) { mnist_images = argv[++a]; }
            else if (!strcmp(argv[a], "--mnist-labels") && a+1 < argc) { mnist_labels = argv[++a]; }
            else if (!strcmp(argv[a], "--limit") && a+1 < argc) { limit = atoi(argv[++a]); }
            else if (!strcmp(argv[a], "--norm") && a+1 < argc) { norm = argv[++a]; }
            else if (!strcmp(argv[a], "--help")) { usage(); return 0; }
        }
        if (units_cnt != layers_hidden) { fprintf(stderr, "units count (%d) must match --layers (%d)\n", units_cnt, layers_hidden); return 1; }
        if (val_frac < 0.0f) val_frac = 0.0f; if (val_frac > 0.9f) val_frac = 0.9f;

        printf("[train] ds=%s layers=%d units=", dataset, layers_hidden);
        for (int i=0;i<units_cnt;i++) printf("%s%d", (i?",":""), units_arr[i]);
        printf(" epochs=%d lr=%.4f seed=%u val=%.2f\n", epochs, lr, seed, val_frac);

        // Dataset + fixed split
        Dataset d = {0};

        if (!strncmp(dataset, "csv:", 4)) {
            const char *path = dataset + 4;
            if (dataset_load_csv(path, csv_has_header, &d) != 0) {
                fprintf(stderr, "CSV load failed: %s\n", path); return 1;
            }
            if (!strcmp(norm, "minmax")) dataset_normalize_minmax(&d);
        } else if (!strcmp(dataset, "mnist")) {
            if (!mnist_images || !mnist_labels) {
                fprintf(stderr, "For --dataset mnist, provide --mnist-images and --mnist-labels (uncompressed IDX).\n");
                return 1;
            }
            if (dataset_load_mnist_idx(mnist_images, mnist_labels, limit, &d) != 0) {
                int rc = dataset_load_mnist_idx(mnist_images, mnist_labels, limit, &d);
                if (rc != 0) { explain_mnist_rc(rc); return 1; }
            }
            // already scaled to [0,1] in loader
        } else if (!strcmp(dataset, "xor") || !strcmp(dataset, "and") || !strcmp(dataset, "or")) {
            d = load_dataset(dataset); // existing synthetic
        } else {
            fprintf(stderr, "Unknown dataset: %s\n", dataset);
            return 1;
        }

        int *idx_all = (int*)malloc((size_t)d.n * sizeof(int));
        if (!idx_all) { fprintf(stderr, "out of memory for index array\n"); dataset_free(&d); return 1; }

        for (int i = 0; i < d.n; ++i) idx_all[i] = i;
        shuffle_indices(idx_all, d.n, seed);

        int n_val = (int)floorf(val_frac * (float)d.n);
        if (n_val < 0) n_val = 0;
        if (n_val > d.n - 1) n_val = d.n - 1; // keep at least 1 train example
        int n_train = d.n - n_val;

        int *idx_val   = idx_all;          // first n_val
        int *idx_train = idx_all + n_val;  // remainder

        // Model + grads
        MLP m = (MLP){0};
        if (mlp_init(&m, /*d_in=*/d.d, /*d_out=*/d.k, units_arr, units_cnt, seed) != 0) {
            fprintf(stderr, "mlp_init failed\n"); dataset_free(&d); return 1;
        }

        // Grad buffers aligned to layers
        Tensor *dW = (Tensor*)malloc(m.L * sizeof(Tensor));
        Tensor *db = (Tensor*)malloc(m.L * sizeof(Tensor));
        for (int l=0;l<m.L;l++){ dW[l]=tensor_alloc(m.W[l].rows,m.W[l].cols); db[l]=tensor_alloc(1,m.b[l].cols); }

        Tensor x = tensor_alloc(1, m.d_in);
        Tensor logits = tensor_alloc(1, m.d_out);


        double t0 = now_ms();
        for (int e = 1; e <= epochs; ++e) {
            double e0 = now_ms();

            // shuffle train indices only (fixed val set)
            shuffle_indices(idx_train, n_train, seed + (unsigned)e);

            int correct_train = 0;
            float loss_sum = 0.0f;

            // train loop
            for (int t = 0; t < n_train; ++t) {
                int i = idx_train[t];
                // Copy features of sample i into x (1 × d.d)
                const float *row = &d.X[(size_t)i * (size_t)d.d];
                for (int j = 0; j < d.d; ++j) x.data[j] = row[j];

                int y = d.y[i];

                mlp_forward_logits(&m, &x, &logits);
                float L = cross_entropy_from_logits(&logits, y);
                loss_sum += L;

                if (argmax(&logits) == y) correct_train++;

                mlp_backward_from_logits(&m, &x, y, dW, db, /*alpha=*/0.1f);
                mlp_sgd_step(&m, dW, db, lr);
            }

            // validation (if any)
            int correct_val = 0;
            if (n_val > 0) {
                for (int t = 0; t < n_val; ++t) {
                    int i = idx_val[t];
                    const float *row = &d.X[(size_t)i * (size_t)d.d];
                    for (int j = 0; j < d.d; ++j) x.data[j] = row[j];

                    int y = d.y[i];
                    mlp_forward_logits(&m, &x, &logits);
                    if (argmax(&logits) == y) correct_val++;
                }
            }

            double e_ms = now_ms() - e0;
            float acc_tr = (float)correct_train / (float)n_train;
            float acc_va = (n_val>0) ? ((float)correct_val / (float)n_val) : NAN;

            if (n_val > 0) {
                printf("[epoch %3d] loss=%.6f acc=%.2f%% val=%.2f%% time=%.1fms\n",
                    e, loss_sum/(float)n_train, acc_tr*100.0f, acc_va*100.0f, e_ms);
            } else {
                printf("[epoch %3d] loss=%.6f acc=%.2f%% time=%.1fms\n",
                    e, loss_sum/(float)n_train, acc_tr*100.0f, e_ms);
            }

            if (acc_tr >= 0.95f && (n_val == 0 || acc_va >= 0.95f)) {
                puts("[train] reached >=95% accuracy — stopping early.");
                break;
            }
        }
        double t_ms = now_ms() - t0;

        printf("[train] total time: %.1fms\n", t_ms);

        if (out_path) {
            if (io_save_mlp(&m, out_path) == 0) {
                printf("[save] wrote model to %s\n", out_path);
            } else {
                fprintf(stderr, "[save] failed to write %s\n", out_path);
            }
        }

        // cleanup
        tensor_free(&logits); tensor_free(&x);
        for (int l=0;l<m.L;l++){ tensor_free(&db[l]); tensor_free(&dW[l]); }
        free(db); free(dW);
        mlp_free(&m);
        free(idx_all);
        return 0;
    }

    if (strcmp(argv[1], "predict") == 0) {
        const char *model_path = NULL;
        for (int a = 2; a < argc; ++a) {
            if (!strcmp(argv[a], "--model") && a+1 < argc) { model_path = argv[++a]; }
        }
        if (!model_path) { fprintf(stderr, "usage: ./perceptron predict --model path\n"); return 2; }

        MLP m = {0};
        if (io_load_mlp(&m, model_path) != 0) { fprintf(stderr, "load failed: %s\n", model_path); return 1; }

        Dataset d = load_dataset("xor");
        Tensor x = tensor_alloc(1, m.d_in);
        Tensor logits = tensor_alloc(1, m.d_out);

        int correct = 0;
        for (int i = 0; i < d.n; ++i) {
            x.data[0] = d.X[2*i + 0];
            x.data[1] = d.X[2*i + 1];
            mlp_forward_logits(&m, &x, &logits);
            if (argmax(&logits) == d.y[i]) correct++;
        }
        printf("[predict] XOR accuracy: %.2f%% (%d/%d)\n", 100.0f*correct/d.n, correct, d.n);

        tensor_free(&logits); tensor_free(&x);
        mlp_free(&m);
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
