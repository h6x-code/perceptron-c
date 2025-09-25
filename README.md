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
  --val 0.1 \
  --layers 2 --units 128,64 \
  --epochs 30 \
  --lr 0.05 \
  --batch 32 \
  --momentum 0.9 \
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
`[epoch  29] loss=0.146914 acc=99.55% time=4377.5ms`

`[train] reached >=99% accuracy — stopping early.`

`[save] wrote model to data/out/model.bin`

- Model should converge early
- Binary should save to specified location

### valgrind shows no leaks:
```bash
valgrind --leak-check=full ./perceptron train <args>
```

- Expect: All heap blocks were freed -- no leaks are possible.