# perceptron-c

A simple multi-layer perceptron (MLP) written in C.  
Supports training on MNIST data, inference on CSV input, and multi-threaded training.

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
```

---

## CLI Overview

The binary has several subcommands:
- `train` – train a model
- `predict` – run inference from a saved model
- `tensor-test` – test tensor ops
- `nn-test` – test neural net forward pass
- `gradcheck` – gradient check via finite differences

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
Download MNIST (not included in repo), then:
```bash
./perceptron train \
  --dataset mnist \
  --mnist-images data/MNIST/raw/train-images-idx3-ubyte \
  --mnist-labels data/MNIST/raw/train-labels-idx1-ubyte \
  --val 0.1 \
  --layers 2 --units 256,64 \
  --epochs 50 \
  --batch 128 --threads 8 \
  --lr 0.05 --momentum 0.9 \
  --lr-decay 0.9 --lr-step 3 --patience 10 \
  --seed 1337 \
  --out data/out/mnist-2layer.bin
```

---

## Run prediction
```bash
./perceptron predict \
  --model data/out/mnist-2layer.bin \
  --input csv:test.csv --csv-has-header
```

---

## Multithreading performance
### DO NOT RUN THE FOLLOWING WITHOUT ADJUSTING MAX NUMBER OF THREADS FOR YOUR SYSTEM
Generate log files:
```bash
./scripts/bench.sh
```

Print .md table:
```bash
./scripts/parse_bench.py
```
### Mini-Training Performance (10k MNIST samples, 2×128,64 MLP, 5 epochs)
| Threads | Total time (s) | Speedup | Efficiency (%) | Best Val (%) |
|--------:|---------------:|--------:|---------------:|-------------:|
| 1 | 19.90 | 1.00 | 100.0 | 96.00 |
| 2 | 11.10 | 1.79 | 89.6 | 96.00 |
| 3 | 8.10 | 2.46 | 81.9 | 96.00 |
| 4 | 7.00 | 2.84 | 71.1 | 96.00 |
| 5 | 6.10 | 3.26 | 65.2 | 96.00 |
| 6 | 5.70 | 3.49 | 58.2 | 96.00 |
| 7 | 5.30 | 3.75 | 53.6 | 96.00 |
| 8 | 5.20 | 3.83 | 47.8 | 96.00 |
| 9 | 5.00 | 3.98 | 44.2 | 96.00 |
| 10 | 5.10 | 3.90 | 39.0 | 96.00 |
| 11 | 5.20 | 3.83 | 34.8 | 96.00 |
| 12 | 5.40 | 3.69 | 30.7 | 96.00 |
| 13 | 7.40 | 2.69 | 20.7 | 96.00 |
| 14 | 6.50 | 3.06 | 21.9 | 96.00 |
| 15 | 7.30 | 2.73 | 18.2 | 96.00 |
| 16 | 7.30 | 2.73 | 17.0 | 96.00 |

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
- Always check with valgrind --leak-check=full after code changes.
- Use `--seed` for reproducibility.
- Tune `--batch` and `--threads` for your CPU to get best throughput.
