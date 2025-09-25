
#define _POSIX_C_SOURCE 200809L

#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>


// ---------- CSV loader (simple numeric, last column is label) ----------

static int is_blank_line(const char *s) {
    while (*s) { if (!isspace((unsigned char)*s)) return 0; s++; }
    return 1;
}

static int count_csv_rows_cols(FILE *f, int has_header, int *out_rows, int *out_cols) {
    char buf[1<<12];
    int rows = 0, cols = -1;
    long start = ftell(f);
    while (fgets(buf, sizeof(buf), f)) {
        if (is_blank_line(buf)) continue;
        if (has_header && rows == 0) { rows++; continue; } // count header row but skip for data
        int c = 1;
        for (char *p = buf; *p; ++p) if (*p == ',') c++;
        if (cols < 0) cols = c;
    }
    fseek(f, start, SEEK_SET);
    if (cols <= 0) return -1;
    *out_cols = cols;
    *out_rows = rows - (has_header ? 1 : 0);
    return 0;
}

int dataset_load_csv(const char *path, int has_header, Dataset *out_ds) {
    memset(out_ds, 0, sizeof(*out_ds));
    FILE *f = fopen(path, "r");
    if (!f) return 1;

    int rows = 0, cols = 0;
    if (count_csv_rows_cols(f, has_header, &rows, &cols) != 0 || rows <= 0 || cols < 2) {
        fclose(f);
        return 2;
    }

    int n = rows;
    int d = cols - 1;
    float *X = (float*)malloc((size_t)n * (size_t)d * sizeof(float));
    int   *y = (int  *)malloc((size_t)n * sizeof(int));
    if (!X || !y) { fclose(f); free(X); free(y); return 3; }

    char buf[1<<12];
    int row = - (has_header ? 1 : 0);
    int kmax = -1;

    while (fgets(buf, sizeof(buf), f)) {
        if (is_blank_line(buf)) continue;
        row++;
        if (has_header && row < 0) continue;
        int col = 0;
        char *save = NULL;
        char *tok = strtok_r(buf, ",\n\r", &save);
        while (tok && col < cols) {
            if (col < d) {
                X[(size_t)row * d + col] = (float)atof(tok);
            } else {
                int cls = (int)strtol(tok, NULL, 10);
                y[row] = cls;
                if (cls > kmax) kmax = cls;
            }
            col++;
            tok = strtok_r(NULL, ",\n\r", &save);
        }
        if (col != cols) { fclose(f); free(X); free(y); return 4; }
    }
    fclose(f);

    out_ds->n = n; out_ds->d = d; out_ds->k = kmax + 1;
    out_ds->X = X; out_ds->y = y;
    return 0;
}

// ---------- MNIST IDX (uncompressed) ----------

static int read_be32(FILE *f, int *out) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 1;
    *out = (int)((b[0]<<24) | (b[1]<<16) | (b[2]<<8) | b[3]);
    return 0;
}

int dataset_load_mnist_idx(const char *images_path, const char *labels_path,
                           int limit, Dataset *out_ds)
{
    memset(out_ds, 0, sizeof(*out_ds));

    FILE *fi = fopen(images_path, "rb");
    FILE *fl = fopen(labels_path, "rb");
    if (!fi || !fl) { if (fi) fclose(fi); if (fl) fclose(fl); return 1; }

    int magic_i=0, magic_l=0;
    int n_i=0, n_l=0, rows=0, cols=0;

    if (read_be32(fi, &magic_i) || read_be32(fi, &n_i) ||
        read_be32(fi, &rows)    || read_be32(fi, &cols)) { fclose(fi); fclose(fl); return 2; }

    if (read_be32(fl, &magic_l) || read_be32(fl, &n_l)) { fclose(fi); fclose(fl); return 3; }

    if (magic_i != 0x00000803 || magic_l != 0x00000801 || n_i != n_l) {
        fclose(fi); fclose(fl); return 4;
    }

    int n = n_i;
    if (limit > 0 && limit < n) n = limit;
    int d = rows * cols;
    int k = 10;

    float *X = (float*)malloc((size_t)n * (size_t)d * sizeof(float));
    int   *y = (int  *)malloc((size_t)n * sizeof(int));
    if (!X || !y) { fclose(fi); fclose(fl); free(X); free(y); return 5; }

    // read images
    for (int i = 0; i < n; ++i) {
        for (int p = 0; p < d; ++p) {
            int c = fgetc(fi);
            if (c == EOF) { fclose(fi); fclose(fl); free(X); free(y); return 6; }
            X[(size_t)i * d + p] = (float)c / 255.0f; // scale to [0,1]
        }
    }

    // skip any remaining images if limit < n_i
    if (limit > 0 && limit < n_i) fseek(fi, (long)( (long long)(n_i - limit) * d ), SEEK_CUR);

    // read labels
    for (int i = 0; i < n; ++i) {
        int c = fgetc(fl);
        if (c == EOF) { fclose(fi); fclose(fl); free(X); free(y); return 7; }
        y[i] = c;
    }

    fclose(fi); fclose(fl);
    out_ds->n = n; out_ds->d = d; out_ds->k = k;
    out_ds->X = X; out_ds->y = y;
    return 0;
}

// ---------- Normalization & free ----------

int dataset_normalize_minmax(Dataset *ds) {
    if (!ds || ds->n <= 0 || ds->d <= 0) return 1;
    for (int j = 0; j < ds->d; ++j) {
        float mn = ds->X[j], mx = ds->X[j];
        for (int i = 1; i < ds->n; ++i) {
            float v = ds->X[(size_t)i * ds->d + j];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        float denom = (mx - mn);
        if (denom == 0.0f) denom = 1.0f; // avoid div-by-zero for constant features
        for (int i = 0; i < ds->n; ++i) {
            float *v = &ds->X[(size_t)i * ds->d + j];
            *v = (*v - mn) / denom;
        }
    }
    return 0;
}

// For raw byte-like features 0..255 → [0,1] (no-op for already scaled)
int dataset_scale_01(Dataset *ds) {
    if (!ds || ds->n <= 0 || ds->d <= 0) return 1;
    for (int i = 0; i < ds->n * ds->d; ++i) {
        ds->X[i] = ds->X[i] / 255.0f;
    }
    return 0;
}

void dataset_free(Dataset *ds) {
    if (!ds) return;
    free(ds->X); ds->X = NULL;
    free(ds->y); ds->y = NULL;
    ds->n = ds->d = ds->k = 0;
}
