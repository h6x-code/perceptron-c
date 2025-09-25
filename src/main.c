#include <math.h>
#include <pthread.h>
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
#include "thread.h"

// Forward declarations to avoid implicit decls:
static int argmax(const Tensor *t);
static void usage(void);

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

// Context for workers
typedef struct {
    const MLP *m;
    const Dataset *din;
    int *pred;
    float *conf;
} PredCtx;

// Multithreading structs
typedef struct {
    MLPWS ws;
    Tensor x, logits;
    Tensor *dW_local, *db_local; // arrays of length m.L
    float loss; int correct;
} ThreadSlot;

typedef struct {
    int start, end, t0;              // shard in current batch
    const int *idx;
    const Dataset *d;
    const MLP *m;
    ThreadSlot *slot;
} Slice;

// Multithreading helpers
static void alloc_param_like(const MLP *m, Tensor **dW, Tensor **db) {
    *dW = (Tensor*)malloc((size_t)m->L * sizeof(Tensor));
    *db = (Tensor*)malloc((size_t)m->L * sizeof(Tensor));
    for (int l = 0; l < m->L; ++l) {
        (*dW)[l] = tensor_alloc(m->W[l].rows, m->W[l].cols);
        (*db)[l] = tensor_alloc(1, m->b[l].cols);
    }
}

static void zero_param_stack(Tensor *dW, Tensor *db, int L) {
    for (int l = 0; l < L; ++l) { tensor_zero(&dW[l]); tensor_zero(&db[l]); }
}

static void free_param_stack(Tensor *dW, Tensor *db, int L) {
    for (int l = 0; l < L; ++l) { tensor_free(&dW[l]); tensor_free(&db[l]); }
    free(dW); free(db);
}

static void* train_slice_run(void *arg) {
    Slice *s = (Slice*)arg;
    ThreadSlot *S = s->slot;
    S->loss = 0.0f; S->correct = 0;
    zero_param_stack(S->dW_local, S->db_local, S->ws.L);

    for (int u = s->start; u < s->end; ++u) {
        int i = s->idx[s->t0 + u];
        const float *row = &s->d->X[(size_t)i * (size_t)s->d->d];
        for (int j = 0; j < s->m->d_in; ++j) S->x.data[j] = row[j];
        int y = s->d->y[i];

        mlp_forward_logits_ws(s->m, &S->x, &S->logits, &S->ws);
        S->loss += cross_entropy_from_logits(&S->logits, y);
        if (argmax(&S->logits) == y) S->correct++;

        mlp_backward_from_logits_ws(s->m, y, &S->ws, S->dW_local, S->db_local, 0.1f);
    }
    return NULL;
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

static void alloc_like_params(const MLP *m, Tensor **W, Tensor **b) {
    *W = (Tensor*)malloc(m->L * sizeof(Tensor));
    *b = (Tensor*)malloc(m->L * sizeof(Tensor));
    for (int l = 0; l < m->L; ++l) {
        (*W)[l] = tensor_alloc(m->W[l].rows, m->W[l].cols);
        (*b)[l] = tensor_alloc(1, m->b[l].cols);
    }
}

static void free_params(Tensor *W, Tensor *b, int L) {
    for (int l = 0; l < L; ++l) { tensor_free(&W[l]); tensor_free(&b[l]); }
    free(W); free(b);
}

static void copy_params(Tensor *dstW, Tensor *dstB, const Tensor *srcW, const Tensor *srcB, int L) {
    for (int l = 0; l < L; ++l) {
        int nW = srcW[l].rows * srcW[l].cols;
        int nb = srcB[l].cols;
        memcpy(dstW[l].data, srcW[l].data, (size_t)nW * sizeof(float));
        memcpy(dstB[l].data, srcB[l].data, (size_t)nb * sizeof(float));
    }
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

static void predict_range(int start, int end, void *user) {
    PredCtx *C = (PredCtx*)user;
    Tensor x = tensor_alloc(1, C->m->d_in);
    Tensor logits = tensor_alloc(1, C->m->d_out);

    for (int i = start; i < end; ++i) {
        const float *row = &C->din->X[(size_t)i * (size_t)C->din->d];
        for (int j = 0; j < C->m->d_in; ++j) x.data[j] = row[j];

        // forward (stateless wrt shared model: only reads W,b; uses local x,logits)
        mlp_forward_logits(C->m, &x, &logits);

        // softmax → confidence
        float mx = logits.data[0];
        for (int k = 1; k < C->m->d_out; ++k) if (logits.data[k] > mx) mx = logits.data[k];
        float sum = 0.f;
        for (int k = 0; k < C->m->d_out; ++k) { logits.data[k] = expf(logits.data[k] - mx); sum += logits.data[k]; }
        int best = 0; float bestp = logits.data[0] / sum;
        for (int k = 1; k < C->m->d_out; ++k) {
            float p = logits.data[k] / sum;
            if (p > bestp) { bestp = p; best = k; }
        }
        C->pred[i] = best;
        C->conf[i] = bestp;
    }
    tensor_free(&logits);
    tensor_free(&x);
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0) {
        usage();
        return 0;
    }

    if (strcmp(argv[1], "train") == 0) {
        int epochs = 100;
        float  lr = 0.01f;
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
        int batch = 32; // mini-batch size
        float momentum = 0.0f;  // 0 = vanilla SGD
        float lr_decay = 1.0f;  //no decay by default
        int lr_step = 0;    // 0 = never
        int patience = 0;   // 0 = disabled
        int threads = 1;

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
            else if (!strcmp(argv[a], "--batch") && a+1 < argc)    { batch = atoi(argv[++a]); }
            else if (!strcmp(argv[a], "--momentum") && a+1 < argc) { momentum = (float)atof(argv[++a]); }
            else if (!strcmp(argv[a], "--help")) { usage(); return 0; }
            else if (!strcmp(argv[a], "--lr-decay") && a+1 < argc) { lr_decay = (float)atof(argv[++a]); }
            else if (!strcmp(argv[a], "--lr-step")  && a+1 < argc) { lr_step  = atoi(argv[++a]); }
            else if (!strcmp(argv[a], "--patience") && a+1 < argc) { patience = atoi(argv[++a]); }
            else if (!strcmp(argv[a], "--threads") && a+1 < argc) { threads = atoi(argv[++a]); if (threads < 1) threads = 1; }
        }
        if (units_cnt != layers_hidden) { fprintf(stderr, "units count (%d) must match --layers (%d)\n", units_cnt, layers_hidden); return 1; }
        if (val_frac < 0.0f) val_frac = 0.0f; if (val_frac > 0.9f) val_frac = 0.9f;

        printf("[train] ds=%s layers=%d units=", dataset, layers_hidden);
        for (int i=0;i<units_cnt;i++) printf("%s%d", (i?",":""), units_arr[i]);
        printf(" epochs=%d lr=%.4f seed=%u val=%.2f batch=%d mom=%.2f decay=%.3f step=%d patience=%d\n", epochs, lr, seed, val_frac, batch, momentum, lr_decay, lr_step, patience);

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

        // Allocate per-thread slots
        int T = threads;
        ThreadSlot *th = (ThreadSlot*)calloc((size_t)T, sizeof(ThreadSlot));
        for (int t = 0; t < T; ++t) {
            mlpws_init(&th[t].ws, &m);
            th[t].x      = tensor_alloc(1, m.d_in);
            th[t].logits = tensor_alloc(1, m.d_out);
            alloc_param_like(&m, &th[t].dW_local, &th[t].db_local);
        }

        // Grad buffers aligned to layers
        Tensor *dW = (Tensor*)malloc(m.L * sizeof(Tensor));
        Tensor *db = (Tensor*)malloc(m.L * sizeof(Tensor));
        for (int l=0;l<m.L;l++){ dW[l]=tensor_alloc(m.W[l].rows,m.W[l].cols); db[l]=tensor_alloc(1,m.b[l].cols); }

        Tensor *dW_tmp = (Tensor*)malloc(m.L * sizeof(Tensor));
        Tensor *db_tmp = (Tensor*)malloc(m.L * sizeof(Tensor));
        for (int l = 0; l < m.L; ++l) {
            dW_tmp[l] = tensor_alloc(m.W[l].rows, m.W[l].cols);
            db_tmp[l] = tensor_alloc(1, m.b[l].cols);
        }

        Tensor x = tensor_alloc(1, m.d_in);
        Tensor logits = tensor_alloc(1, m.d_out);

        SGD_Momentum optm = {0};
        int use_momentum = (momentum > 0.0f);
        if (use_momentum) {
            if (sgd_momentum_init(&optm, m.L, m.W, m.b, momentum) != 0) {
                fprintf(stderr, "momentum init failed\n"); /* free & exit */ }
        }

        float best_metric = -1.0f;    // best val acc (or train acc if no val)
        int   since_improve = 0;

        Tensor *bestW = NULL, *bestB = NULL;
        if (out_path) {                      // only snapshot if we plan to save
            alloc_like_params(&m, &bestW, &bestB);
            copy_params(bestW, bestB, m.W, m.b, m.L); // initial snapshot
            best_metric = 0.0f;
        }

        double t0 = now_ms();
        for (int e = 1; e <= epochs; ++e) {
            double e0 = now_ms();

            // shuffle train indices only (fixed val set)
            shuffle_indices(idx_train, n_train, seed + (unsigned)e);

            int correct_train = 0;
            float loss_sum = 0.0f;

            // train loop (multi-threading)
            for (int t0 = 0; t0 < n_train; t0 += batch) {
                int B = batch; if (t0 + B > n_train) B = n_train - t0;
                int TT = T; if (TT > B) TT = B;

                // partition [0,B) → TT slices
                int base = B / TT, rem = B % TT, cur = 0;
                Slice *S = (Slice*)malloc((size_t)TT * sizeof(Slice));
                pthread_t *pth = (pthread_t*)malloc((size_t)TT * sizeof(pthread_t));
                for (int t = 0; t < TT; ++t) {
                    int len = base + (t < rem ? 1 : 0);
                    S[t] = (Slice){ .start=cur, .end=cur+len, .t0=t0, .idx=idx_train, .d=&d, .m=&m, .slot=&th[t] };
                    cur += len;
                    pthread_create(&pth[t], NULL, train_slice_run, &S[t]);
                }
                for (int t = 0; t < TT; ++t) pthread_join(pth[t], NULL);

                // Reduce per-thread grads deterministically with Kahan in double, and average by B
                Tensor *dW = NULL, *db = NULL;
                alloc_param_like(&m, &dW, &db);

                const double invB = 1.0 / (double)B;

                // For each layer, reduce weights then biases
                for (int l = 0; l < m.L; ++l) {
                    const int nW = m.W[l].rows * m.W[l].cols;
                    const int nb = m.b[l].cols;

                    // Weights: Kahan compensated sum in fixed thread order (0..TT-1)
                    for (int i = 0; i < nW; ++i) {
                        double sum = 0.0, c = 0.0;
                        for (int t = 0; t < TT; ++t) {
                            double y = (double)th[t].dW_local[l].data[i] - c;
                            double tmp = sum + y;
                            c = (tmp - sum) - y;
                            sum = tmp;
                        }
                        dW[l].data[i] = (float)(sum * invB);
                    }

                    // Biases: Kahan compensated sum in fixed thread order (0..TT-1)
                    for (int j = 0; j < nb; ++j) {
                        double sum = 0.0, c = 0.0;
                        for (int t = 0; t < TT; ++t) {
                            double y = (double)th[t].db_local[l].data[j] - c;
                            double tmp = sum + y;
                            c = (tmp - sum) - y;
                            sum = tmp;
                        }
                        db[l].data[j] = (float)(sum * invB);
                    }
                }

                // Single optimizer step with averaged grads
                if (use_momentum) sgd_momentum_step(&optm, m.W, m.b, dW, db, lr);
                else              sgd_step_params  (m.W, m.b, dW, db, m.L, lr);
                free_param_stack(dW, db, m.L);

                // batch stats
                int corr = 0; float lsum = 0.0f;
                for (int t = 0; t < TT; ++t) { corr += th[t].correct; lsum += th[t].loss; }
                correct_train += corr; loss_sum += lsum;

                free_param_stack(dW, db, m.L);
                free(pth); free(S);
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

            double e_s = (now_ms() - e0) / 1000.0;
            float acc_tr = (float)correct_train / (float)n_train;
            float acc_va = (n_val>0) ? ((float)correct_val / (float)n_val) : NAN;

            // Choose the metric: prefer validation if present
            float metric = (n_val > 0) ? acc_va : acc_tr;

            // Update best & early stopping
            if (metric > best_metric + 1e-7f) {
                best_metric = metric;
                since_improve = 0;
                if (bestW) copy_params(bestW, bestB, m.W, m.b, m.L);
            } else {
                since_improve++;
            }

            // Learning-rate decay
            if (lr_step > 0 && (e % lr_step) == 0 && lr_decay > 0.0f && lr_decay < 1.0f) {
                lr *= lr_decay;
                if (lr < 1e-8f) lr = 1e-8f; // safety floor
                printf("[train] lr decayed to %.6f\n", lr);
            }

            // Early stop (only if patience > 0; if no val, patience uses train metric)
            if (patience > 0 && since_improve >= patience) {
                printf("[train] early stop (no improvement for %d epochs). Best=%.2f%%\n",
                    patience, best_metric * 100.0f);
                break;
            }

            if (n_val > 0) {
                printf("[epoch %3d] loss=%.6f acc=%.2f%% val=%.2f%% time=%.1fs\n",
                    e, loss_sum/(float)n_train, acc_tr*100.0f, acc_va*100.0f, e_s);
            } else {
                printf("[epoch %3d] loss=%.6f acc=%.2f%% time=%.1fs\n",
                    e, loss_sum/(float)n_train, acc_tr*100.0f, e_s);
            }

            if (acc_tr >= 0.99f && (n_val == 0 || acc_va >= 0.99f)) {
                puts("[train] reached >=99% accuracy — stopping early.");
                break;
            }
        }
        double t_ms = now_ms() - t0;
        double t_s = t_ms / 1000.0;

        printf("[train] total time: %.1fs\n", t_s);

        // restore best snapshot if available
        if (bestW && bestB) {
            copy_params(m.W, m.b, bestW, bestB, m.L);
        }

        if (out_path) {
            if (io_save_mlp(&m, out_path) == 0) {
                printf("[save] wrote model to %s\n", out_path);
            } else {
                fprintf(stderr, "[save] failed to write %s\n", out_path);
            }
        }

        // cleanup
        tensor_free(&logits);
        tensor_free(&x);
        for (int l = 0; l < m.L; ++l) { tensor_free(&db[l]); tensor_free(&dW[l]); }
        free(db);
        free(dW);
        mlp_free(&m);
        if (use_momentum) sgd_momentum_free(&optm);
        if (bestW) { free_params(bestW, bestB, m.L); }

        // Free multithreading
        for (int t = 0; t < T; ++t) {
            free_param_stack(th[t].dW_local, th[t].db_local, m.L);
            tensor_free(&th[t].logits);
            tensor_free(&th[t].x);
            mlpws_free(&th[t].ws);
        }
        free(th);


        // free indices + dataset
        free(idx_all);
        dataset_free(&d);

        return 0;

    }

    if (strcmp(argv[1], "predict") == 0) {
        const char *model_path = NULL;
        const char *input_spec = NULL;   // still supports csv:...
        int csv_has_header = 0;
        int threads = 1;

        // MNIST IDX flags
        const char *mnist_images = NULL;
        const char *mnist_labels = NULL;
        int limit = 0; // 0 = no limit

        for (int a = 2; a < argc; ++a) {
            if      (!strcmp(argv[a], "--model") && a+1 < argc)         { model_path = argv[++a]; }
            else if (!strcmp(argv[a], "--input") && a+1 < argc)         { input_spec = argv[++a]; }
            else if (!strcmp(argv[a], "--csv-has-header"))              { csv_has_header = 1; }
            else if (!strcmp(argv[a], "--threads") && a+1 < argc)       { threads = atoi(argv[++a]); if (threads < 1) threads = 1; }
            // NEW:
            else if (!strcmp(argv[a], "--mnist-images") && a+1 < argc)  { mnist_images = argv[++a]; }
            else if (!strcmp(argv[a], "--mnist-labels") && a+1 < argc)  { mnist_labels = argv[++a]; }
            else if (!strcmp(argv[a], "--limit") && a+1 < argc)         { limit = atoi(argv[++a]); if (limit < 0) limit = 0; }
            else if (!strcmp(argv[a], "--help")) { usage(); return 0; }
        }
        if (!model_path) { fprintf(stderr, "predict: require --model\n"); return 1; }

        // Load model
        MLP m = {0};
        if (io_load_mlp(&m, model_path) != 0) {
            fprintf(stderr, "[predict] failed to load model: %s\n", model_path);
            return 1;
        }

        // Load inputs
        Dataset din = {0};
        int have_csv = (input_spec && !strncmp(input_spec, "csv:", 4));
        int have_mnist = (mnist_images != NULL);

        if (have_csv && have_mnist) {
            fprintf(stderr, "[predict] choose EITHER --input csv:... OR --mnist-images/--mnist-labels\n");
            mlp_free(&m); return 1;
        }

        if (have_csv) {
            const char *path = input_spec + 4;
            int rc = dataset_load_csv_features(path, csv_has_header, &din);
            if (rc != 0) { fprintf(stderr, "[predict] CSV load failed (%d): %s\n", rc, path); mlp_free(&m); return 1; }
        } else if (have_mnist) {
            if (!mnist_labels) {
                fprintf(stderr, "[predict] for MNIST, provide both --mnist-images and --mnist-labels\n");
                mlp_free(&m); return 1;
            }
            int rc = dataset_load_mnist_idx(mnist_images, mnist_labels, limit, &din);
            if (rc != 0) {
                fprintf(stderr, "[predict] MNIST IDX load failed (%d): images=%s labels=%s\n", rc, mnist_images, mnist_labels);
                mlp_free(&m); return 1;
            }
        } else {
            fprintf(stderr, "[predict] provide --input csv:... OR --mnist-images/--mnist-labels\n");
            mlp_free(&m); return 1;
        }

        if (din.d != m.d_in) {
            fprintf(stderr, "[predict] feature dim mismatch: model d_in=%d, input d=%d\n", m.d_in, din.d);
            dataset_free(&din); mlp_free(&m); return 1;
        }


        // Threaded inference
        const int n = din.n;
        if (threads > n) threads = n > 0 ? n : 1;

        int   *pred  = (int*)  malloc((size_t)n * sizeof(int));
        float *conf  = (float*)malloc((size_t)n * sizeof(float));
        if (!pred || !conf) { fprintf(stderr, "[predict] OOM\n"); free(pred); free(conf); dataset_free(&din); mlp_free(&m); return 1; }

        PredCtx ctx = { .m = &m, .din = &din, .pred = pred, .conf = conf };

        printf("[predict] n=%d d=%d -> k=%d threads=%d\n", din.n, din.d, m.d_out, threads);

        if (parallel_for(n, threads, predict_range, &ctx) != 0) {
            fprintf(stderr, "[predict] threading failed\n");
            free(conf); free(pred); dataset_free(&din); mlp_free(&m); return 1;
        }

        int correct = 0;
        for (int i = 0; i < n; ++i) {
            printf("%d,%.6f", pred[i], conf[i]);
            if (din.y) { // labels present (MNIST path)
                printf(",%d", din.y[i]);
                if (pred[i] == din.y[i]) correct++;
            }
            printf("\n");
        }
        if (din.y) {
            float acc = (n > 0) ? (100.0f * (float)correct / (float)n) : 0.0f;
            fprintf(stderr, "[predict] accuracy=%.2f%% (%d/%d)\n", acc, correct, n);
        }

        free(conf);
        free(pred);
        dataset_free(&din);
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
