#!/usr/bin/env python3
"""
Plot perceptron-c training curves from log files.

Parses lines like:
  [epoch   7] loss=0.021822 acc=99.30% val=98.17% time=20.2s
  [train] lr decayed to 0.047500
  [train] total time: 584.6s

Usage:
  python3 scripts/plot_results.py path/to/log1.txt [path/to/log2.txt ...]
    --out plots/curves.png
    --title "MNIST MLP (threads=8)"
    --no-val             # hide validation accuracy curve
    --no-lr              # hide LR curve
    --dpi 150
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

EPOCH_RE = re.compile(
    r"\[epoch\s*(\d+)\]\s+loss=([0-9.+-eE]+)\s+acc=([0-9.+-eE]+)%"
    r"(?:\s+val=([0-9.+-eE]+)%)?.*?(?:time=([0-9.+-eE]+)s)?"
)

LR_RE = re.compile(r"\[train\]\s+lr decayed to\s+([0-9.+-eE]+)")
TOTAL_TIME_RE = re.compile(r"\[train\]\s+total time:\s*([0-9.+-eE]+)s")

def parse_log(path: Path):
    epochs, loss, acc, val, lr_points = [], [], [], [], []
    total_time = None
    cur_epoch_lr = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                ep = int(m.group(1))
                l = float(m.group(2))
                a = float(m.group(3))
                v = float(m.group(4)) if m.group(4) is not None else None
                epochs.append(ep)
                loss.append(l)
                acc.append(a)
                val.append(v)
                # If we saw an LR before (from previous "[train] lr decayed..." line),
                # associate the most recent LR with this epoch.
                if cur_epoch_lr:
                    # grab the last known lr
                    last_lr_epoch = max(cur_epoch_lr.keys())
                    lr_points.append((ep, cur_epoch_lr[last_lr_epoch]))
                continue

            m = LR_RE.search(line)
            if m:
                lr = float(m.group(1))
                # Store LR keyed by "next epoch we encounter"; will attach when we see an epoch
                # (We still also show the last known LR on the legend.)
                if epochs:
                    lr_points.append((epochs[-1], lr))  # mark change at the last recorded epoch
                else:
                    cur_epoch_lr[0] = lr  # before first epoch, harmless
                continue

            m = TOTAL_TIME_RE.search(line)
            if m:
                total_time = float(m.group(1))

    # Build dense LR per epoch if we have points
    lr_curve = None
    if lr_points:
        # We may have multiple entries per epoch; keep the last for that epoch
        by_ep = {}
        for ep, lr in lr_points:
            by_ep[ep] = lr
        # Fill forward: start with first known LR, then carry last seen
        lr_curve = []
        last = None
        epoch_min = min(epochs) if epochs else 1
        epoch_max = max(epochs) if epochs else 0
        for ep in range(epoch_min, epoch_max + 1):
            if ep in by_ep:
                last = by_ep[ep]
            lr_curve.append(last)

    return {
        "file": str(path),
        "epochs": epochs,
        "loss": loss,
        "acc": acc,
        "val": val,
        "lr": lr_curve,
        "total_time": total_time,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="Log file(s) produced by ./perceptron")
    ap.add_argument("--out", default="plots/training_curves.png", help="Output image path (png/pdf/svg)")
    ap.add_argument("--title", default="Training curves", help="Figure title")
    ap.add_argument("--no-val", action="store_true", help="Hide validation accuracy")
    ap.add_argument("--no-lr", action="store_true", help="Hide LR secondary axis")
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    args = ap.parse_args()

    runs = [parse_log(Path(p)) for p in args.logs]

    # Make figure
    plt.figure(figsize=(9, 5.5))

    # Left y-axis: Loss (line), Accuracy (right side shared x but separate scale)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Distinguish runs by label (filename stem)
    for run in runs:
        label_base = Path(run["file"]).stem
        if run["epochs"]:
            ax1.plot(run["epochs"], run["loss"], linestyle="-", marker="", label=f"loss ({label_base})")

            ax2.plot(run["epochs"], run["acc"], linestyle="--", marker="", label=f"acc ({label_base})")

            if not args.no_val:
                # Plot only where val is present
                v_epochs = [e for e, v in zip(run["epochs"], run["val"]) if v is not None]
                v_vals = [v for v in run["val"] if v is not None]
                if v_epochs:
                    ax2.plot(v_epochs, v_vals, linestyle=":", marker="", label=f"val ({label_base})")

            if run["lr"] and (not args.no_lr):
                # Secondary x/y for LR on ax1 right? Better: overlay as dotted thin line with second y-axis
                ax3 = ax1.twinx()
                # Offset LR axis to the right so it doesn't clash with acc axis
                ax3.spines.right.set_position(("axes", 1.08))
                ax3.set_frame_on(True)
                ax3.patch.set_visible(False)
                ax3.plot(range(run["epochs"][0], run["epochs"][-1] + 1), run["lr"], linestyle="-.", marker="", label=f"lr ({label_base})")
                ax3.set_ylabel("Learning rate")
                # Combine legends later
        else:
            print(f"[warn] No epochs parsed from {run['file']}")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy (%)")
    plt.title(args.title)

    # Build a single legend by collecting from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # If an LR axis exists, capture its labels too (there can be many if multiple runs)
    lr_labels = []
    lr_handles = []
    for ax in plt.gcf().axes:
        if ax is not ax1 and ax is not ax2:
            h, l = ax.get_legend_handles_labels()
            lr_handles.extend(h)
            lr_labels.extend(l)

    handles = handles1 + handles2 + lr_handles
    labels = labels1 + labels2 + lr_labels
    if handles:
        plt.legend(handles, labels, loc="best", fontsize=9)

    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi)
    print(f"[plot] wrote {out_path}")

if __name__ == "__main__":
    main()
