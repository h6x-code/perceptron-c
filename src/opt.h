#pragma once

// Simple SGD step: param[i] -= lr * grad[i] for i in [0, n)
static inline void sgd_step(float *param, const float *grad, int n, float lr) {
    for (int i = 0; i < n; i++) {
        param[i] -= lr * grad[i];
    }
}
