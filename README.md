# perceptron-c
Building a modular perceptron in C for educational purposes.

## testing
compile from project root 
```bash
make clean
make
```
train
```bash
./perceptron train --dataset xor --layers 1 --units 4 --epochs 500 --lr 0.1 --seed 1337 --out data/out/model.bin
```
check these for back
```bash
# sanity: previous commands still work
./perceptron --help
./perceptron train xor
./perceptron tensor-test 1337
./perceptron nn-test
./perceptron loss-test
./perceptron gradcheck
```

## Success criteria (expected output)

### ./perceptron train <args> prints something like:
`[epoch  44] loss=0.631788 acc=100.00%`
`[train] reached >=95% accuracy — stopping early.`
`[save] wrote model to data/out/model.bin`

- Model should converge early
- Binary should save to specified location

### valgrind shows no leaks:
```bash
valgrind --leak-check=full ./perceptron train <args>
```

- Expect: All heap blocks were freed -- no leaks are possible.