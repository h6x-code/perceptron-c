#!/usr/bin/env python3
import sys, re, glob, os
from pathlib import Path

LOG_GLOB = sys.argv[1] if len(sys.argv) > 1 else "logs/*.log"

# Patterns
re_total = re.compile(r"\[train\]\s+total time:\s*([0-9]*\.?[0-9]+)\s*(ms|s)")
re_val   = re.compile(r"val=([0-9]*\.?[0-9]+)%")
re_acc   = re.compile(r"acc=([0-9]*\.?[0-9]+)%")
re_threads_from_name = re.compile(r"thread(\d+)", re.IGNORECASE)

def to_seconds(val, unit):
    return float(val) / 1000.0 if unit == "ms" else float(val)

def parse_log(path: str):
    """Return dict with: threads, total_s, best_val, last_acc."""
    threads = None
    m = re_threads_from_name.search(os.path.basename(path))
    if m:
        threads = int(m.group(1))

    total_s = None
    best_val = None
    last_acc = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # track best validation accuracy seen
            vm = re_val.search(line)
            if vm:
                v = float(vm.group(1))
                best_val = v if best_val is None else max(best_val, v)
            # keep last reported train acc (in case no val is present)
            am = re_acc.search(line)
            if am:
                last_acc = float(am.group(1))
            # capture total time
            tm = re_total.search(line)
            if tm:
                total_s = to_seconds(tm.group(1), tm.group(2))

    return {
        "file": path,
        "threads": threads,
        "total_s": total_s,
        "best_val": best_val,
        "last_acc": last_acc,
    }

def fmt(x, nd=2):
    return f"{x:.{nd}f}"

def main():
    files = sorted(glob.glob(LOG_GLOB))
    if not files:
        print(f"No logs matched: {LOG_GLOB}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for fp in files:
        d = parse_log(fp)
        if d["threads"] is None:
            # fallback: use index if no threadN in filename
            d["threads"] = len(rows) + 1
        rows.append(d)

    # Keep only logs that have total time
    rows = [r for r in rows if r["total_s"] is not None]
    if not rows:
        print("No valid logs (no total time found).", file=sys.stderr)
        sys.exit(2)

    # Sort by thread count
    rows.sort(key=lambda r: r["threads"])

    # Baseline for speedup
    base = None
    for r in rows:
        if r["threads"] == 1:
            base = r["total_s"]
            break
    if base is None:
        # fall back to smallest threads as baseline
        base = rows[0]["total_s"]

    # Build Markdown table
    print("| Threads | Total time (s) | Speedup | Efficiency (%) | Best Val (%) |")
    print("|--------:|---------------:|--------:|---------------:|-------------:|")
    for r in rows:
        t = r["threads"]
        tot = r["total_s"]
        sp = base / tot if tot and base else float("nan")
        eff = 100.0 * sp / t if t and sp == sp else float("nan")
        best_val = r["best_val"] if r["best_val"] is not None else (r["last_acc"] if r["last_acc"] is not None else float("nan"))
        print(f"| {t} | {fmt(tot, 2)} | {fmt(sp, 2)} | {fmt(eff, 1)} | {fmt(best_val, 2) if best_val==best_val else '—'} |")

    # Optional: show which files were parsed
    print("\n_Parsed logs:_")
    for r in rows:
        print(f"- {Path(r['file']).as_posix()}")

if __name__ == "__main__":
    main()
