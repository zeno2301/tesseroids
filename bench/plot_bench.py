#!/usr/bin/env python3
"""
Plot CPU vs CUDA benchmark results.

Reads a CSV file with columns:
    prisms,points,field,cpu_seconds,cuda_seconds
(as produced by bench/bench_prisms.py --csv results.csv) and draws:

  1. time vs number of prisms (log-log), CPU and CUDA curves
  2. speedup (CPU time / CUDA time) vs number of prisms

Usage:
    python3 bench/plot_bench.py results.csv [--out bench.png] [--field gz]
"""
import argparse
import csv
import sys


def load(path, field=None):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if field and r["field"] != field:
                continue
            rows.append({
                "prisms": int(r["prisms"]),
                "points": int(r["points"]),
                "field": r["field"],
                "cpu": float(r["cpu_seconds"]) if r["cpu_seconds"] else None,
                "cuda": float(r["cuda_seconds"]) if r["cuda_seconds"] else None,
            })
    rows.sort(key=lambda r: r["prisms"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="bench.png")
    ap.add_argument("--field", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load(args.csv, args.field)
    if not rows:
        print("no rows to plot", file=sys.stderr)
        return 1

    prisms = [r["prisms"] for r in rows]
    cpu = [r["cpu"] for r in rows]
    cuda = [r["cuda"] for r in rows]
    speedup = [(r["cpu"]/r["cuda"] if r["cpu"] and r["cuda"] else None)
               for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if any(cpu):
        ax.plot(prisms, cpu, "o-", label="CPU (bin/)", color="tab:blue")
    if any(cuda):
        ax.plot(prisms, cuda, "s-", label="CUDA (bin-cuda/)", color="tab:orange")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of prisms")
    ax.set_ylabel(f"wall time [s]  ({rows[0]['points']} points, prism{rows[0]['field']})")
    ax.set_title("Forward modelling time: CPU vs CUDA")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    xs = [p for p, s in zip(prisms, speedup) if s is not None]
    ys = [s for s in speedup if s is not None]
    if xs:
        ax.plot(xs, ys, "D-", color="tab:green")
        ax.axhline(1, ls="--", color="gray", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("number of prisms")
    ax.set_ylabel("speedup  (CPU time / CUDA time)")
    ax.set_title("CUDA speedup")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved plot to {args.out}")

    # also print a small table
    print(f"\n{'prisms':>10} {'CPU s':>10} {'CUDA s':>10} {'speedup':>8}")
    for r, s in zip(rows, speedup):
        cpu_s = f"{r['cpu']:.4g}" if r["cpu"] else "-"
        cuda_s = f"{r['cuda']:.4g}" if r["cuda"] else "-"
        sp = f"{s:.1f}x" if s else "-"
        print(f"{r['prisms']:>10} {cpu_s:>10} {cuda_s:>10} {sp:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
