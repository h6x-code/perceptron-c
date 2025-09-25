#pragma once
typedef struct {
    const float *X;  // pointer to n*d floats
    const int   *y;  // pointer to n labels
    int n, d, k;
} Dataset;

Dataset load_dataset(const char *name);
