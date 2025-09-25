#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>

typedef struct {
    uint32_t magic;    // 'P''C''M''L' = 0x4C4D4350
    uint32_t version;  // 1
    uint32_t L;        // number of layers (incl. output)
    uint32_t d_in;
    uint32_t d_out;
} ModelHeader;

static const uint32_t MAGIC_PCML = 0x4C4D4350u; // 'PCML'
static const uint32_t VERSION     = 1u;

// Create parent directory (e.g., "data/out" for "data/out/model.bin")
// Returns 0 on success or if already exists, nonzero on failure.
static int ensure_parent_dir(const char *path) {
    // find last '/'
    const char *slash = strrchr(path, '/');
    if (!slash) return 0; // no parent
    size_t len = (size_t)(slash - path);
    if (len == 0) return 0; // root-level like "/file.bin" (skip here)

    char *dir = (char*)malloc(len + 1);
    if (!dir) return 1;
    memcpy(dir, path, len);
    dir[len] = '\0';

    // Try to mkdir the whole chain (simple "mkdir -p" loop)
    char *p = dir;
    while (*p) {
        if (*p == '/') {
            *p = '\0';
            if (dir[0] != '\0') mkdir(dir, 0755); // ignore errors (already exists)
            *p = '/';
        }
        p++;
    }
    mkdir(dir, 0755); // final component
    free(dir);
    return 0;
}

int io_save_mlp(const MLP *m, const char *path) {
    (void)ensure_parent_dir(path);

    FILE *f = fopen(path, "wb");
    if (!f) return 1;

    ModelHeader h = { MAGIC_PCML, VERSION, (uint32_t)m->L,
                      (uint32_t)m->d_in, (uint32_t)m->d_out };
    if (fwrite(&h, sizeof(h), 1, f) != 1) { fclose(f); return 2; }

    for (int i = 0; i <= m->L; ++i) {
        uint32_t di = (uint32_t)m->dims[i];
        if (fwrite(&di, sizeof(di), 1, f) != 1) { fclose(f); return 3; }
    }
    for (int l = 0; l < m->L; ++l) {
        int nW = m->W[l].rows * m->W[l].cols;
        int nb = m->b[l].cols;
        if ((int)fwrite(m->W[l].data, sizeof(float), nW, f) != nW) { fclose(f); return 4; }
        if ((int)fwrite(m->b[l].data, sizeof(float), nb, f) != nb) { fclose(f); return 5; }
    }
    fclose(f);
    return 0;
}

int io_load_mlp(MLP *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 1;

    ModelHeader h;
    if (fread(&h, sizeof(h), 1, f) != 1) { fclose(f); return 2; }
    if (h.magic != MAGIC_PCML)           { fclose(f); return 3; }
    if (h.version != VERSION)            { fclose(f); return 4; }

    int L = (int)h.L;
    int *dims = (int*)malloc((size_t)(L + 1) * sizeof(int));
    if (!dims) { fclose(f); return 5; }
    for (int i = 0; i <= L; ++i) {
        uint32_t di;
        if (fread(&di, sizeof(di), 1, f) != 1) { free(dims); fclose(f); return 6; }
        dims[i] = (int)di;
    }

    // init empty model using dims from file
    int d_in  = dims[0];
    int d_out = dims[L];
    if (mlp_init(m, d_in, d_out, /*hidden*/ dims + 1, /*n_hidden*/ L - 1, /*seed unused*/ 0) != 0) {
        free(dims); fclose(f); return 7;
    }

    // read weights/biases
    for (int l = 0; l < L; ++l) {
        int nW = m->W[l].rows * m->W[l].cols;
        int nb = m->b[l].cols;
        if ((int)fread(m->W[l].data, sizeof(float), nW, f) != nW) { free(dims); fclose(f); mlp_free(m); return 8; }
        if ((int)fread(m->b[l].data, sizeof(float), nb, f) != nb) { free(dims); fclose(f); mlp_free(m); return 9; }
    }

    free(dims);
    fclose(f);
    return 0;
}
