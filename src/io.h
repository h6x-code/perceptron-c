#pragma once
#include <stdint.h>
#include "nn_model.h"

// Save/load MLP to a compact binary file.
// Returns 0 on success, nonzero on failure.
int io_save_mlp(const MLP *m, const char *path);
int io_load_mlp(MLP *m, const char *path);  // mlp_free(m) before reusing
