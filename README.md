# perceptron-c
Building a classic 0-9 digit perceptron in C for educational purposes.

## testing 
```bash
# compile from project root
make clean
make

./perceptron gradcheck

# sanity: previous commands still work
./perceptron --help
./perceptron train xor
./perceptron tensor-test 1337
./perceptron nn-test
./perceptron loss-test
```

## Success criteria (expected output)

### ./perceptron gradcheck prints something like:
`[loss] before=0.802xxx after=0.561xxx (expect after < before)`

- For loss-test, after must be strictly less than before.

### valgrind shows no leaks:
```bash
valgrind --leak-check=full ./perceptron loss-test
```

- Expect: All heap blocks were freed -- no leaks are possible.