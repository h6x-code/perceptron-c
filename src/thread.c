#include "thread.h"
#include <pthread.h>
#include <stdlib.h>

typedef struct { int start, end; ParForBody body; void *ctx; } Task;

static void *worker(void *arg) {
    Task *t = (Task*)arg;
    t->body(t->start, t->end, t->ctx);
    return NULL;
}

int parallel_for(int n, int num_threads, ParForBody body, void *ctx) {
    if (n <= 0 || num_threads <= 1) { body(0, n, ctx); return 0; }
    if (num_threads > n) num_threads = n;

    pthread_t *ths = (pthread_t*)malloc((size_t)num_threads * sizeof(pthread_t));
    Task      *tsk = (Task*)     malloc((size_t)num_threads * sizeof(Task));
    if (!ths || !tsk) { free(ths); free(tsk); return 1; }

    int base = n / num_threads, rem = n % num_threads, cur = 0;
    for (int t = 0; t < num_threads; ++t) {
        int len = base + (t < rem ? 1 : 0);
        tsk[t] = (Task){ .start = cur, .end = cur + len, .body = body, .ctx = ctx };
        cur += len;
        pthread_create(&ths[t], NULL, worker, &tsk[t]);
    }
    for (int t = 0; t < num_threads; ++t) pthread_join(ths[t], NULL);
    free(tsk); free(ths);
    return 0;
}
