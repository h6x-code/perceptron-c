#pragma once
#include <stddef.h>

typedef struct {
    int n;      // samples
    int d;      // features per sample
    int k;      // number of classes
    float *X;   // size n*d (row-major: sample i at X + i*d)
    int   *y;   // size n
} Dataset;

// existing: synthetic datasets
Dataset load_dataset(const char *name);  // "and", "or", "xor"

// NEW: CSV loader (last column is label). If has_header!=0, skip first line.
int dataset_load_csv(const char *path, int has_header, Dataset *out_ds);

// Load a CSV with ONLY features (no label column). If has_header!=0 skip first line.
// Fills out_ds->n and ->d and allocates X; sets k=0 and y=NULL.
int dataset_load_csv_features(const char *path, int has_header, Dataset *out_ds);

// NEW: MNIST IDX loader (expects uncompressed idx3-ubyte/idx1-ubyte)
int dataset_load_mnist_idx(const char *images_path, const char *labels_path,
                           int limit, Dataset *out_ds);

// Normalization (in-place). Returns 0 on success.
int dataset_normalize_minmax(Dataset *ds);       // per feature to [0,1]
int dataset_scale_01(Dataset *ds);               // assume 0..255 -> divide by 255
void dataset_free(Dataset *ds);
