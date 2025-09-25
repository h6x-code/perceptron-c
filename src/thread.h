#pragma once
#include <stddef.h>

typedef void (*ParForBody)(int start, int end, void *ctx);

// Runs body over [0, n) split across num_threads. Returns 0 on success.
int parallel_for(int n, int num_threads, ParForBody body, void *ctx);
