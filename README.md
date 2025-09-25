# perceptron-c
Building a classic 0-9 digit perceptron in C for educational purposes.

## testing 
```bash
# compile from project root
clang -std=c11 -O2 -Wall -Wextra -pedantic src/*.c -o perceptron

# smoke test (hidden)
./perceptron tensor-test 1337

# sanity: previous commands still work
./perceptron --help
./perceptron train xor
```

## Success criteria (expected output)

### ./perceptron tensor-test [seed] prints something like:
```
[tensor] 2x3 sum_zero=0.0 sum_rand=0.123
```

- sum_zero must be exactly 0.0.

- sum_rand should be non-zero and deterministic given the seed.

### valgrind shows no leaks:
```bash
valgrind --leak-check=full ./perceptron tensor-test 1337
```

Expect: All heap blocks were freed -- no leaks are possible.