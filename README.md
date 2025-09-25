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
./perceptron train \
  --dataset mnist \
  --mnist-images data/MNIST/raw/train-images-idx3-ubyte \
  --mnist-labels data/MNIST/raw/train-labels-idx1-ubyte \
  --limit 2000 --val 0.1 \
  --layers 1 --units 64 \
  --epochs 10 \
  --lr 0.01 \
  --seed 1337 \
  --out data/out/model.bin
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
`[epoch   5] loss=0.146914 acc=95.55% time=4377.5ms`
`[train] reached >=95% accuracy — stopping early.`
`[save] wrote model to data/out/model.bin`

- Model should converge early
- Binary should save to specified location

### valgrind shows no leaks:
```bash
valgrind --leak-check=full ./perceptron train <args>
```

- Expect: All heap blocks were freed -- no leaks are possible.