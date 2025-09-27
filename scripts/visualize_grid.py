#!/usr/bin/env python3
"""
visualize_grid.py

Quick visualizations for grid search results produced by scripts/grid_search.sh.

Input CSV is expected to have (at least) columns:
  model_key,threads,layers,units,batch,lr,momentum,decay,step,epochs,
  train_time_s,best_val_pct,test_acc_pct,model_path,log_path

Outputs:
  - PNG plots:
      speed_vs_accuracy_all.png
      speed_vs_accuracy_pareto.png
      best_acc_per_thread.png
      median_time_per_thread.png
  - CSV leaderboards in outdir/csv/:
      top_by_test_acc.csv
      top_by_speed.csv
      pareto_frontier.csv
      per_thread_best.csv
      per_thread_median_time.csv

Usage:
  python3 visualize_grid.py runs/grid_*/results.csv --out runs/vis
"""

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib
import matplotlib.pyplot as plt


# -------------------------------
# Parsing & helpers
# -------------------------------

def _to_float(val, default=float("nan")) -> float:
    try:
        if val is None or val == "":
            return default
        return float(val)
    except Exception:
        return default


def _to_int(val, default=None):
    try:
        if val is None or val == "":
            return default
        return int(val)
    except Exception:
        return default


def read_results(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            print(f"[warn] missing CSV: {p}", file=sys.stderr)
            continue
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            r = csv.DictReader(f)
            for row in r:
                # Normalize types
                row["threads"] = _to_int(row.get("threads"))
                row["layers"] = _to_int(row.get("layers"))
                row["batch"] = _to_int(row.get("batch"))
                row["lr"] = _to_float(row.get("lr"))
                row["momentum"] = _to_float(row.get("momentum"))
                row["decay"] = _to_float(row.get("decay"))
                row["step"] = _to_int(row.get("step"))
                row["epochs"] = _to_int(row.get("epochs"))
                row["train_time_s"] = _to_float(row.get("train_time_s"))
                row["best_val_pct"] = _to_float(row.get("best_val_pct"))
                row["test_acc_pct"] = _to_float(row.get("test_acc_pct"))
                rows.append(row)
    return rows


def ensure_out(outdir: Path) -> Tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    csvdir = outdir / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    return outdir, csvdir


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


# -------------------------------
# Metrics & aggregations
# -------------------------------

def add_speed_columns(rows: List[Dict[str, Any]]) -> None:
    """Add speedup/efficiency columns to rows, using best T=1 time when available,
    else global best time as a fallback (not ideal, but still informative)."""
    t1_times = [r["train_time_s"] for r in rows if r.get("threads") == 1 and not math.isnan(r.get("train_time_s", float("nan")))]
    if len(t1_times) > 0:
        baseline = min(t1_times)
        baseline_note = "T=1 min"
    else:
        baseline = min([r["train_time_s"] for r in rows if not math.isnan(r.get("train_time_s", float("nan")))])
        baseline_note = "global min"
    for r in rows:
        tt = r.get("train_time_s")
        th = r.get("threads") or 1
        if tt and tt > 0:
            r["speedup"] = baseline / tt
            r["efficiency_pct"] = 100.0 * r["speedup"] / max(1, th)
        else:
            r["speedup"] = float("nan")
            r["efficiency_pct"] = float("nan")
    print(f"[info] speedup baseline: {baseline:.3f}s ({baseline_note})")


def pareto_frontier(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute Pareto frontier on (min train_time_s, max test_acc_pct)."""
    filt = [r for r in rows if not math.isnan(r.get("train_time_s", float("nan"))) and not math.isnan(r.get("test_acc_pct", float("nan")))]
    # Sort by time asc, then keep rows that improve accuracy
    filt.sort(key=lambda r: (r["train_time_s"], -r["test_acc_pct"]))
    frontier = []
    best_acc = -1.0
    for r in filt:
        acc = r["test_acc_pct"]
        if acc > best_acc:
            frontier.append(r)
            best_acc = acc
    return frontier


def per_thread_best(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_t: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        t = r.get("threads")
        if t is None:
            continue
        by_t.setdefault(t, []).append(r)
    best = []
    for t, lst in by_t.items():
        # Best by test acc, fallback to best val, fallback to fastest
        valid = [x for x in lst if not math.isnan(x.get("test_acc_pct", float("nan")))]
        if not valid:
            valid = [x for x in lst if not math.isnan(x.get("best_val_pct", float("nan")))]
            key = lambda x: x.get("best_val_pct", float("nan"))
        else:
            key = lambda x: x.get("test_acc_pct", float("nan"))
        if valid:
            row = max(valid, key=key)
        else:
            row = min(lst, key=lambda x: x.get("train_time_s", float("inf")))
        best.append(row)
    best.sort(key=lambda r: r.get("threads", 0))
    return best


def per_thread_median_time(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    import statistics as stats
    by_t: Dict[int, List[float]] = {}
    for r in rows:
        t = r.get("threads")
        tt = r.get("train_time_s")
        if t is None or math.isnan(tt):
            continue
        by_t.setdefault(t, []).append(tt)
    out = []
    for t, times in by_t.items():
        med = stats.median(times) if times else float("nan")
        out.append({"threads": t, "median_train_time_s": med})
    out.sort(key=lambda r: r["threads"])
    return out


# -------------------------------
# Plotting (matplotlib; no explicit colors)
# -------------------------------

def fig_save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=144)
    plt.close(fig)
    print(f"[plot] wrote {path}")


def plot_speed_vs_accuracy(rows: List[Dict[str, Any]], out: Path, title="Speed vs Accuracy", show=False):
    xs = [r.get("train_time_s") for r in rows]
    ys = [r.get("test_acc_pct") for r in rows]
    labels = [r.get("model_key", "") for r in rows]
    th = [r.get("threads", 1) for r in rows]

    # Scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, alpha=0.7, s=24)

    ax.set_xlabel("Train time (s)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)

    # Annotate a few most-interesting (fastest 3, and top-3 accuracy)
    try:
        # fastest 3
        order_fast = sorted(range(len(rows)), key=lambda i: xs[i] if not math.isnan(xs[i]) else float("inf"))[:3]
        for i in order_fast:
            ax.annotate(f"fast#{i+1}", (xs[i], ys[i]))
        # highest acc 3
        order_acc = sorted(range(len(rows)), key=lambda i: -(ys[i] if not math.isnan(ys[i]) else -float("inf")))[:3]
        for i in order_acc:
            ax.annotate(f"acc#{i+1}", (xs[i], ys[i]))
    except Exception:
        pass

    fig_save(fig, out)
    if show:
        plt.show()


def plot_speed_vs_accuracy_pareto(rows, outpath: Path, title="Speed vs Accuracy (Pareto)", show=False):
    """
    Scatter plot of train_time_s vs test_acc_pct with the Pareto frontier highlighted.
    """
    import math
    import matplotlib.pyplot as plt

    # Only keep rows with both metrics present and finite
    clean = [r for r in rows
             if r.get("train_time_s") is not None
             and r.get("test_acc_pct") is not None
             and not math.isnan(r["train_time_s"])
             and not math.isnan(r["test_acc_pct"])]

    if not clean:
        print(f"[warn] no valid points to plot for Pareto: {outpath}")
        return

    # Compute Pareto frontier on the cleaned rows
    frontier = pareto_frontier(clean, time_key="train_time_s", acc_key="test_acc_pct")

    # Unzip points
    xs = [r["train_time_s"] for r in clean]
    ys = [r["test_acc_pct"] for r in clean]

    fx = [r["train_time_s"] for r in frontier]
    fy = [r["test_acc_pct"] for r in frontier]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, alpha=0.5, s=22, label="Configs")
    ax.plot(fx, fy, "-o", linewidth=2, markersize=4, label="Pareto frontier")

    ax.set_xlabel("Train time (s)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath)
    print(f"[plot] wrote {outpath}")
    if show:
        plt.show()
    plt.close(fig)



def plot_best_acc_per_thread(rows: List[Dict[str, Any]], out: Path, title="Best accuracy per thread", show=False):
    best = per_thread_best(rows)
    xs = [r["threads"] for r in best]
    ys = [r.get("test_acc_pct") if not math.isnan(r.get("test_acc_pct", float("nan"))) else r.get("best_val_pct") for r in best]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(xs, ys)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Best accuracy (%)")
    ax.set_title(title)
    fig_save(fig, out)
    if show:
        plt.show()


def plot_median_time_per_thread(rows: List[Dict[str, Any]], out: Path, title="Median train time per thread", show=False):
    med = per_thread_median_time(rows)
    xs = [r["threads"] for r in med]
    ys = [r["median_train_time_s"] for r in med]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(xs, ys)
    ax.set_xlabel("Threads")
    ax.set_ylabel("Median train time (s)")
    ax.set_title(title)
    fig_save(fig, out)
    if show:
        plt.show()


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize grid search results.")
    ap.add_argument("results", nargs="+", help="Paths to results.csv (one or more). Glob expansion is handled by your shell.")
    ap.add_argument("--out", default="runs/vis", help="Output directory for plots/CSVs.")
    ap.add_argument("--title", default="Grid search: speed vs accuracy", help="Plot title base.")
    ap.add_argument("--show", action="store_true", help="Show plots interactively.")
    args = ap.parse_args()

    paths = [Path(p) for p in args.results]
    rows = read_results(paths)
    if not rows:
        print("[error] No rows parsed. Check your input CSV paths.", file=sys.stderr)
        sys.exit(2)

    add_speed_columns(rows)

    outdir, csvdir = ensure_out(Path(args.out))

    # Leaderboards
    # Top by test accuracy (fallback to best_val if test_acc is missing)
    have_test = [r for r in rows if not math.isnan(r.get("test_acc_pct", float("nan")))]
    if have_test:
        top_by_acc = sorted(have_test, key=lambda r: r["test_acc_pct"], reverse=True)[:20]
    else:
        valid_val = [r for r in rows if not math.isnan(r.get("best_val_pct", float("nan")))]
        top_by_acc = sorted(valid_val, key=lambda r: r["best_val_pct"], reverse=True)[:20]

    top_by_speed = sorted([r for r in rows if not math.isnan(r.get("train_time_s", float("nan")))],
                          key=lambda r: r["train_time_s"])[:20]

    frontier = pareto_frontier(rows)
    # per_t_best = per_thread_best(rows)
    # med_time = per_thread_median_time(rows)

    # Write CSVs
    def_fields = ["model_key","threads","layers","units","batch","lr","momentum","decay","step","epochs",
                  "train_time_s","speedup","efficiency_pct","best_val_pct","test_acc_pct","model_path","log_path"]
    write_csv(csvdir / "all_results.csv", rows, def_fields)
    write_csv(csvdir / "top_by_test_acc.csv", top_by_acc, def_fields)
    write_csv(csvdir / "top_by_speed.csv", top_by_speed, def_fields)
    write_csv(csvdir / "pareto_frontier.csv", frontier, def_fields)
    # write_csv(csvdir / "per_thread_best.csv", per_t_best, def_fields)
    # write_csv(csvdir / "per_thread_median_time.csv", med_time, ["threads","median_train_time_s"])

    # Plots
    plot_speed_vs_accuracy(rows, outdir / "speed_vs_accuracy_all.png", title=args.title, show=args.show)
    plot_speed_vs_accuracy_pareto(rows, outdir / "speed_vs_accuracy_pareto.png", title=args.title + " (Pareto)", show=args.show)
    # plot_best_acc_per_thread(rows, outdir / "best_acc_per_thread.png", show=args.show)
    # plot_median_time_per_thread(rows, outdir / "median_time_per_thread.png", show=args.show)

    print(f"[done] wrote outputs under {outdir}")


if __name__ == "__main__":
    main()
