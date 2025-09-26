#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from glob import glob

import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"""^\[epoch\s*(\d+)\]\s*loss=([0-9.]+)\s*acc=([0-9.]+)%(?:\s*val=([0-9.]+)%)?""",
    re.I,
)
TIME_RE = re.compile(r"\[train\]\s*total time:\s*([0-9.]+)s", re.I)


def parse_log(path: Path):
    """
    Parse a single log file:
      - label: first token that looks like '<n> threads' found in file name or header line
      - epoch rows: epoch, loss, acc, (val_acc)
      - total_time: seconds
      - best_val: max val accuracy seen
    """
    label = path.stem
    # Try to recover thread-count style labels from filename (e.g., thread8.log or threads8.log)
    m = re.search(r"(\d+)\s*threads", label)
    if not m:
        m = re.search(r"threads?(\d+)", label)
    if m:
        label = f"{m.group(1)} threads"

    epochs, loss, acc, val = [], [], [], []
    total_time = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                e = int(m.group(1))
                l = float(m.group(2))
                a = float(m.group(3))
                v = float(m.group(4)) if m.group(4) is not None else None
                epochs.append(e)
                loss.append(l)
                acc.append(a)
                if v is not None:
                    val.append(v)
                continue
            m = TIME_RE.search(line)
            if m:
                total_time = float(m.group(1))

    best_val = max(val) if val else None
    return {
        "label": label,
        "epochs": epochs,
        "loss": loss,
        "acc": acc,
        "val": val if val else None,
        "total_time": total_time,
        "best_val": best_val,
        "threads": extract_threads_from_label(label),
    }


def extract_threads_from_label(label: str):
    m = re.search(r"(\d+)\s*threads", label)
    return int(m.group(1)) if m else None


def ema(xs, alpha):
    if alpha <= 0:
        return xs[:]
    out = []
    s = None
    for x in xs:
        if s is None:
            s = x
        else:
            s = alpha * x + (1 - alpha) * s
        out.append(s)
    return out


def plot_runs(
    runs,
    out_path,
    title,
    ylim_loss=None,
    ylim_acc=None,
    smooth=0.0,
    legend_outside=False,
    speed_chart="none",  # "none", "time", "speedup", "both"
):
    # Determine subplot rows
    show_speed = speed_chart in ("time", "speedup", "both")
    nrows = 3 if show_speed else 2

    fig = plt.figure(figsize=(14, 9 if show_speed else 7))
    gs = fig.add_gridspec(nrows=nrows, ncols=1, height_ratios=[1, 1] + ([0.7] if show_speed else []), hspace=0.35)

    # ----- Loss (top) -----
    ax_loss = fig.add_subplot(gs[0, 0])
    for r in runs:
        if not r["epochs"]:
            continue
        y = ema(r["loss"], smooth) if smooth > 0 else r["loss"]
        ax_loss.plot(r["epochs"], y, label=f"{r['label']}")
    ax_loss.set_title(title)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    if ylim_loss:
        ax_loss.set_ylim(ylim_loss)

    # ----- Accuracy (middle) -----
    ax_acc = fig.add_subplot(gs[1, 0])
    for r in runs:
        if not r["epochs"]:
            continue
        y = ema(r["acc"], smooth) if smooth > 0 else r["acc"]
        ax_acc.plot(r["epochs"], y, label=f"{r['label']} (train)")
        if r["val"]:
            yv = ema(r["val"], smooth) if smooth > 0 else r["val"]
            ax_acc.plot(r["epochs"][: len(yv)], yv, linestyle="--", label=f"{r['label']} (val)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid(True, alpha=0.3)
    if ylim_acc:
        ax_acc.set_ylim(ylim_acc)

    # Legend placement
    if legend_outside:
        ax_acc.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    else:
        ax_acc.legend()

    # ----- Speed chart (optional bottom) -----
    if show_speed:
        ax_spd = fig.add_subplot(gs[2, 0])
        # sort by threads if available; else keep insertion order
        sruns = sorted(runs, key=lambda r: (r["threads"] is None, r["threads"] if r["threads"] is not None else 0))
        xs, ys, labels = [], [], []
        for r in sruns:
            if r["threads"] is None or r["total_time"] is None:
                continue
            xs.append(r["threads"])
            if speed_chart in ("time", "both"):
                ys.append(r["total_time"])
            labels.append(r["label"])

        if speed_chart in ("time", "both"):
            ax_spd.plot(xs, ys, marker="o")
            ax_spd.set_ylabel("Total time (s)")
            ax_spd.set_xlabel("Threads")
            ax_spd.grid(True, alpha=0.3)

        if speed_chart in ("speedup", "both"):
            # compute speedups against run["speedup"] pre-filled in main()
            xs2, su = [], []
            for r in sruns:
                if r["threads"] is None or r.get("speedup") is None:
                    continue
                xs2.append(r["threads"])
                su.append(r["speedup"])
            if speed_chart == "both":
                ax_spd2 = ax_spd.twinx()
                ax_spd2.plot(xs2, su, marker="s", linestyle="--")
                ax_spd2.set_ylabel("Speedup (×)")
                ax_spd2.grid(False)
            else:
                ax_spd.cla()
                ax_spd.plot(xs2, su, marker="o", linestyle="-")
                ax_spd.set_xlabel("Threads")
                ax_spd.set_ylabel("Speedup (×)")
                ax_spd.grid(True, alpha=0.3)

    # ----- Footer summary (times / best-val / speedup) -----
    footer = []
    for r in runs:
        frag = r["label"]
        mode = r.get("_footer_mode", "all")
        if mode in ("all", "time") and r["total_time"] is not None:
            frag += f"  time={r['total_time']:.1f}s"
        if mode in ("all", "val") and r["best_val"] is not None:
            frag += f"  best-val={r['best_val']:.2f}%"
        if mode in ("all", "speedup") and r.get("speedup") is not None:
            frag += f"  speedup={r['speedup']:.2f}×"
        footer.append(frag)
    if footer:
        fig.text(0.5, 0.015, " | ".join(footer), ha="center", va="bottom", fontsize=9)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"[plot] wrote {out_path}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="Plot loss/accuracy curves (and optional speed) from perceptron-c logs.")
    ap.add_argument("logs", nargs="+", help="log files or glob patterns (e.g. logs/thread*.log)")
    ap.add_argument("--out", help="output image path (PNG)")
    ap.add_argument("--title", default="Training Curves")
    ap.add_argument("--ylim-loss", type=float, nargs=2, metavar=("LOW", "HIGH"),
                    help="y-limits for loss axis")
    ap.add_argument("--ylim-acc", type=float, nargs=2, metavar=("LOW", "HIGH"),
                    help="y-limits for accuracy axis")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="EMA smoothing alpha (0<alpha<1). Example: 0.2")
    ap.add_argument("--legend-outside", action="store_true",
                    help="place the combined legend outside the lower plot")

    # Speed plotting controls
    ap.add_argument("--speed-chart", choices=["none", "time", "speedup", "both"],
                    default="none", help="add a bottom panel for speed/time or speedup")
    ap.add_argument("--baseline", default="1 threads",
                    help='label of the baseline run to compute speedups (default: "1 threads")')

    # Footer summary content
    ap.add_argument("--footer", choices=["all", "time", "val", "speedup", "none"],
                    default="all", help="what to show in the footer summary (default: all)")

    args = ap.parse_args()

    expanded = []
    for p in args.logs:
        matches = glob(p)
        if matches:
            expanded.extend(matches)
        else:
            print(f"[warn] no files matched: {p}")
    if not expanded:
        raise SystemExit("[error] no input logs after glob expansion.")
    paths = [Path(p) for p in expanded]

    runs = [parse_log(p) for p in paths]

    # Baseline for speedup
    base = next((r for r in runs if r["label"] == args.baseline), None)
    base_time = base["total_time"] if base else None
    for r in runs:
        r["speedup"] = (base_time / r["total_time"]) if (base_time and r["total_time"]) else None
        r["_footer_mode"] = args.footer

    plot_runs(
        runs,
        args.out,
        args.title,
        ylim_loss=tuple(args.ylim_loss) if args.ylim_loss else None,
        ylim_acc=tuple(args.ylim_acc) if args.ylim_acc else None,
        smooth=args.smooth,
        legend_outside=args.legend_outside,
        speed_chart=args.speed_chart,
    )


if __name__ == "__main__":
    main()
