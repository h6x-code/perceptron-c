# perceptron-c

A modular multi-layer perceptron (MLP) written in C.  
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

## Dataset

MNIST: https://github.com/cvdfoundation/mnist
You’ll need the original MNIST IDX files. Place them under `data/MNIST/raw/` and extract the files. You need raw files not `.gz`.

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
  --epochs 40 \
  --batch 128 --threads  \
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
### Mini-Training Performance (10k MNIST samples, 2×256,64 MLP, 10 epochs)
| Threads | Total time (s) | Speedup | Efficiency (%) | Best Val (%) |
|--------:|---------------:|--------:|---------------:|-------------:|
| 1 | 35.80 | 1.00 | 100.0 | 96.60 |
| 2 | 21.10 | 1.70 | 84.8 | 96.60 |
| 4 | 12.80 | 2.80 | 69.9 | 96.60 |
| 8 | 9.60 | 3.73 | 46.6 | 96.60 |
| 16 | 18.00 | 1.99 | 12.4 | 96.60 |

The helper script `scripts/plot_results.py` can visualize training logs.
```bash
python3 scripts/plot_results.py logs/thread*.log \
  --out plots/all_threads.png \
  --title "All thread counts"
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
- Always check with valgrind --leak-check=full after code changes.
- Use `--seed` for reproducibility.
- Tune `--batch` and `--threads` for your CPU to get best throughput.
