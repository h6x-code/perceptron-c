# Perceptron-C (ML Sandbox)

A modular multi-layer perceptron (MLP) written in C.  
Supports training on MNIST data, inference on CSV input, and multi-threaded training.

Built as an educational project: the code emphasizes clarity, memory safety (Valgrind clean), and understanding of how forward/backward propagation, cross-entropy, and SGD actually work at a low level.

To make use of `scripts/`, please see the `README.md` located in that folder. `scripts/` contains `.sh`, `.py`, and `.ipynb` files used to analyze model training and results.

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

## Dataset

MNIST: https://github.com/cvdfoundation/mnist
You’ll need the original MNIST IDX files. Place them under `data/MNIST/raw/` and extract the files. You cannot use `.gz`.

---

## Quick Start

Clone and build:

```bash
git clone https://github.com/h6x-code/perceptron-c.git
cd perceptron-c
make
```

---

## CLI Overview

The binary has several subcommands:
- `train`: train a model
- `predict`: run inference from a saved model
- `eval`: evaluate accuracy on a testing dataset

### Core flags:
- `--dataset xor|and|or|mnist|csv:...`
- `--layers N` and `--units a,b,c`: network architecture
- `--batch B`: mini-batch size
- `--threads`: number of threads for training
- `--epochs E` `--lr α` `--momentum μ`: optimizer hyperparameters
- `--lr-decay r` `--lr-step k`: learning rate scheduling
- `--patience p`: early stopping based on validation accuracy
- `--out path/to/model.bin`: for saving
- `--seed S`: deterministic runs

Run `./perceptron --help` for details.

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
Download MNIST (not included in repo), adjust hyperparameters for your experiment, then:
```bash
./perceptron train \
  --dataset mnist \
  --mnist-images data/MNIST/raw/train-images-idx3-ubyte \
  --mnist-labels data/MNIST/raw/train-labels-idx1-ubyte \
  --limit 10000 --val 0.1 \
  --layers 2 --units 256,32 \
  --epochs 60 \
  --batch 256 --threads 8 \
  --lr 0.1 --momentum 0.92 \
  --lr-decay 0.98 --lr-step 4 --patience 10 \
  --seed 1337 \
  --out data/out/model.bin
```

---

## Run prediction
```bash
./perceptron predict \
  --model data/out/model.bin \
  --input csv:test.csv --csv-has-header
```

Or, for MNIST test set:
```bash
./perceptron predict \
  --model data/out/model.bin \
  --mnist-images data/MNIST/raw/t10k-images-idx3-ubyte \
  --mnist-labels data/MNIST/raw/t10k-labels-idx1-ubyte
```

---

## Evaluate model
Evaluate a trained model (loss, accuracy) on a dataset:
```bash
./perceptron eval \
  --model data/out/model.bin \
  --mnist-images data/MNIST/raw/t10k-images-idx3-ubyte \
  --mnist-labels data/MNIST/raw/t10k-labels-idx1-ubyte
```

---

## Additional Commands

### Run Unit Tests
```bash
./perceptron tensor-test   # smoke test tensor ops
./perceptron nn-test       # verify forward pass
./perceptron gradcheck     # gradient finite-diff check
```

---

## Tips
- Always check with `valgrind --leak-check=full` after code changes.
- Use `--seed` for reproducibility.
- Tune `--batch` and `--threads` for your CPU to get best throughput.
