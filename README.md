# perceptron-c

A from-scratch multilayer perceptron (MLP) in pure C11.  
No external libraries — just `clang`/`gcc`, math, and grit.

Built as an educational project: the code emphasizes clarity, memory safety (Valgrind clean), and understanding of how forward/backward propagation, cross-entropy, and SGD actually work at a low level.

---

## Features

- Written in portable C11 (Linux/macOS).
- Shape-flexible MLP (`--layers` / `--units a,b,c`).
- Forward, backward, cross-entropy, ReLU, softmax.
- Optimizers: SGD + momentum, optional LR scheduling.
- Data loaders: synthetic (XOR, AND, OR), MNIST IDX format.
- Save/load models in compact binary format.
- CLI interface for training and prediction.
- Memory checked with `valgrind`.

---

## Quick Start

Clone and build:

```bash
git clone https://github.com/h6x-code/perceptron-c.git
cd perceptron-c
make
./perceptron tensor-test 42     # smoke test tensors
./perceptron nn-test            # check forward softmax
./perceptron gradcheck          # finite-diff gradient check
```

---

## Training Examples

### XOR (toy dataset)
```bash
./perceptron train \
  --dataset xor \
  --layers 1 --units 4 \
  --epochs 500 \
  --lr 0.1 --seed 1337
```

### MNIST (IDX format)
Download MNIST (not included in repo), then:
```bash
./perceptron train \
  --dataset mnist \
  --mnist-images data/MNIST/raw/train-images-idx3-ubyte \
  --mnist-labels data/MNIST/raw/train-labels-idx1-ubyte \
  --limit 10000 --val 0.1 \
  --layers 1 --units 512 \
  --epochs 50 \
  --lr 0.05 --batch 128 --momentum 0.9 \
  --lr-decay 0.95 --lr-step 3 --patience 10 \
  --seed 1337 \
  --out data/out/model.bin
```

---

## CLI Overview

### Subcommands:
- `train`: train a model
- `predict`: run inference on saved model
- `tensor-test`, `nn-test`, `gradcheck`: internal checks

### Core flags:
- `--layers N` and `--units a,b,c`
- `--epochs E` `--lr α` `--batch B` `--momentum μ`
- `--lr-decay r` `--lr-step k` `--patience p`
- `--dataset xor|and|or|mnist|csv:...`
- `--out path/to/model.bin` (for saving)
- `--seed S` (deterministic runs)
Run `./perceptron --help` for details.