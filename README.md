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
`[gradcheck] loss=0.626060 max_rel_error=7.402e-05`
`[gradcheck] OK (max_rel_error < 1e-4)`

- For loss-test, after must be strictly less than before.

### valgrind shows no leaks:
```bash
valgrind --leak-check=full ./perceptron gradcheck
```

- Expect: All heap blocks were freed -- no leaks are possible.