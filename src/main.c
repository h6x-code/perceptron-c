#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data.h"
#include "nn.h"
#include "tensor.h"

static void usage(void) {
    puts("perceptron - C11 MLP\n"
         "Usage:\n"
         "  ./perceptron --help\n"
         "  ./perceptron train   [flags]\n"
         "  ./perceptron predict [flags]\n"
         "\nDefaults: --layers 1 --units 128 --epochs 10 --lr 0.01 --batch 32 --seed 1337");
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0) {
        usage();
        return 0;
    }
    
    if (strcmp(argv[1], "train") == 0) {
        const char *ds = (argc > 2) ? argv[2] : "xor";
        Dataset d = load_dataset(ds);
        if (d.n == 0) { fprintf(stderr, "unknown dataset '%s'\n", ds); return 1; }
        printf("[train] dataset=%s n=%d d=%d k=%d\n", ds, d.n, d.d, d.k);
        return 0;
    }

    if (strcmp(argv[1], "predict") == 0) {
        puts("[predict] subcommand recognized (flags parsed later).");
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
    fprintf(stderr, "error: unknown subcommand '%s'\n", argv[1]);
    usage();
    return 1;
}
