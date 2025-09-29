# Scripts

Small utilities to run, benchmark, grid-search, and visualize the perceptron trainer.

>All commands assume you run them from the project root (where `./perceptron` lives).

## Contents
- `bench.sh`: quick scaling benchmark (num_threads vs time)
    - meant to run after `bench.sh`:
        - `parse_bench.py`: read logs and print a .md table for quick interpretation
        - `plot_results.py`: visualize logs
- `grid_search.sh`: hyperparameter sweeps + logging + CSV results
    - meant to run after `grid_search.sh`:
        - `visualize_grid.py`: richer plots from a grid serach CSV
- `Hyperparameter-Grid-Search-Analysis.ipynb`: A turnkey notebook for analyzing grid searches

---

## Multithreading performance testing
### DO NOT RUN THE FOLLOWING WITHOUT ADJUSTING MAX NUMBER OF THREADS FOR YOUR SYSTEM
### `bench.sh`
Generate log files:
```bash
./scripts/bench.sh
```

### `parse_bench.py`
Print .md table:
```bash
python3 scripts/parse_bench.py
```
 Example output for Mini-Training Performance (10k MNIST samples, 2×256,64 MLP, 10 epochs):
| Threads | Total time (s) | Speedup | Best Val (%) |
|--------:|---------------:|--------:|-------------:|
| 1 | 35.80 | 1.00 | 96.60 |
| 2 | 21.10 | 1.70 | 96.60 |
| 4 | 12.80 | 2.80 | 96.60 |
| 8 | 9.60 | 3.73 | 96.60 |
| 16 | 18.00 | 1.99 | 96.60 |

### `plot_results.py`
The helper script `scripts/plot_results.py` can visualize training logs.
```bash
# Compare two runs (loss/acc in one figure)
python3 scripts/plot_results.py logs/thread1.log logs/threads8.log \
  --out plots/compare_1_vs_8.png \
  --title "Threads 1 vs 8"

# Plot every log in a folder (shell expands the glob)
python3 scripts/plot_results.py logs/thread*.log \
  --out plots/all_threads.png \
  --title "All thread counts" \
  --ylim-loss 0 0.8 --ylim-acc 80 100 --legend-outside

# Also include a speed vs accuracy scatter
python3 scripts/plot_results.py logs/*.log \
  --speed-out plots/speed_vs_accuracy.png
```

---

## Hyperparameter optimization grid search
### `grid_search.sh`
Run a grid of models, capturing train logs, eval logs, and a consolidated CSV with default grid:
```bash
./scripts/grid_search.sh
```

#### Outputs
Creates a timestamped run folder:
```
runs/grid_YYYYmmdd-HHMMSS/
  ├─ models/    # saved .bin models
  ├─ logs/      # one .train.log per model_key
  ├─ plots/     # optional figures you generate later
  └─ results.csv
```
#### `results.csv` columns
- model_key 
- threads
- layers 
- units
- batch
- lr
- momentum
- decay
- step
- epochs
- train_time_s
- best_val_pct
- test_acc_pct
- model_path
- log_path

### `visualize_grid.py`
Richer plots directly from a grid's `results.csv`.
```bash
# One-stop visualization folder
python3 scripts/visualize_grid.py runs/grid_YYYYmmdd-HHMMSS/results.csv \
  --out runs/grid_YYYYmmdd-HHMMSS/docs

# Customize the title, show interactive windows too
python3 scripts/visualize_grid.py runs/grid_mini/results.csv \
  --out runs/grid_mini/docs --title "Mini grid" --show
```

### `Hyperparameter-Grid-Search-Analysis.ipynb`
Copy notebook to root of the grid
```bash
cp scripts/Hyperparameter-Grid-Search-Analysis.ipynb \
    runs/grid_YYYYmmdd-HHMMSS/Hyperparameter-Grid-Search-Analysis.ipynb
```

Open the notebook and run all cells for analysis.