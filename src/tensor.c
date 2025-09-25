#include "tensor.h"
#include <stdlib.h>
#include <stdint.h>

Tensor tensor_alloc(int r, int c){
    Tensor t = {r, c, NULL};
    size_t n = (size_t)r * (size_t)c;
    t.data = n ? (float*)malloc(n * sizeof(float)) : NULL;
    return t;
}

void tensor_free(Tensor *t){ if(t && t->data){ free(t->data); t->data=NULL; } }

void tensor_zero(Tensor *t){
    size_t n = (size_t)t->rows * (size_t)t->cols;
    for(size_t i=0;i<n;i++) t->data[i]=0.0f;
}

static inline uint32_t lcg(uint32_t *s){ *s = *s*1664525u + 1013904223u; return *s; }
void tensor_randu(Tensor *t, unsigned seed){
    uint32_t s = seed; size_t n = (size_t)t->rows * (size_t)t->cols;
    for(size_t i=0;i<n;i++){ float u = (lcg(&s)>>8) / 16777216.0f; t->data[i]=u*2.0f-1.0f; }
}
