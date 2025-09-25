# perceptron-c
Building a classic 0-9 digit perceptron in C for educational purposes.

## testing 
```bash
# compile from project root
clang -std=c11 -O2 -Wall -Wextra -pedantic src/*.c -o perceptron -lm

./perceptron nn-test

# sanity: previous commands still work
./perceptron --help
./perceptron train xor
./perceptron tensor-test 1337
```

## Success criteria (expected output)

### ./perceptron nn-test prints something like:
`[nn] p=[0...., 0....] sum=1.000000`

- Two probabilities in (0,1).
- Sum is 1.000000 within 1e-6.

### valgrind shows no leaks:
```bash
valgrind --leak-check=full ./perceptron nn-test
```

- Expect: All heap blocks were freed -- no leaks are possible.