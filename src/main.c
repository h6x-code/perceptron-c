#include <stdio.h>
#include <string.h>
#include "data.h"

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
    fprintf(stderr, "error: unknown subcommand '%s'\n", argv[1]);
    usage();
    return 1;
}
