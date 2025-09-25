#include <stdio.h>
#include <string.h>
#include "data.h"
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
    if (strcmp(argv[1], "tensor-test") == 0) {           // dev subcommand
        Tensor t = tensor_alloc(2, 3);
        tensor_zero(&t); float s0=0; for(int i=0;i<6;i++) s0 += t.data[i];
        tensor_randu(&t, 1337); float s1=0; for(int i=0;i<6;i++) s1 += t.data[i];
        printf("[tensor] 2x3 sum_zero=%.1f sum_rand=%.3f\n", s0, s1);
        tensor_free(&t); return 0;
    }
    fprintf(stderr, "error: unknown subcommand '%s'\n", argv[1]);
    usage();
    return 1;
}
