#!/usr/bin/env python3
"""
Benchmark the CPU (bin/) and CUDA (bin-cuda/) prism* programs.

Generates a random prism model and a grid of computation points, then times
both versions of one field program (default: prismgz).

Example:
    python3 bench/bench_prisms.py --prisms 100000 --points 2601 --field gz

CPU numbers are reported only if bin/prism<field> exists; CUDA numbers only if
bin-cuda/prism<field> exists and a GPU is available.
"""
import argparse
import os
import random
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIELDS = ["pot", "gx", "gy", "gz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz"]


def gen_model(path, n, seed=42):
    rng = random.Random(seed)
    with open(path, "w") as f:
        for _ in range(n):
            x1 = rng.uniform(-10000, 10000); dx = rng.uniform(50, 500)
            y1 = rng.uniform(-10000, 10000); dy = rng.uniform(50, 500)
            z1 = rng.uniform(0, 2000);      dz = rng.uniform(50, 500)
            dens = rng.uniform(2000, 3500)
            f.write(f"{x1} {x1+dx} {y1} {y1+dy} {z1} {z1+dz} {dens}\n")


def gen_points(path, n, seed=7):
    # square-ish grid covering the model area
    rng = random.Random(seed)
    side = int(n ** 0.5) + 1
    with open(path, "w") as f:
        for i in range(side):
            for j in range(side):
                if i*side + j >= n:
                    break
                f.write(f"{rng.uniform(-10000, 10000)} "
                        f"{rng.uniform(-10000, 10000)} 2000.0\n")


def time_run(prog, model, points):
    with open(points) as fin:
        t0 = time.time()
        p = subprocess.run([prog, model, "-v"], stdin=fin,
                           capture_output=True, text=True)
        dt = time.time() - t0
    if p.returncode != 0:
        return None, p.stderr.strip()
    return dt, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prisms", type=int, default=100000)
    ap.add_argument("--points", type=int, default=2601)
    ap.add_argument("--field", default="gz", choices=FIELDS)
    args = ap.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="tesseroids_bench_")
    model = os.path.join(tmpdir, "model.txt")
    points = os.path.join(tmpdir, "points.txt")
    print(f"Generating model with {args.prisms} prisms and {args.points} points...")
    gen_model(model, args.prisms)
    gen_points(points, args.points)

    print(f"Benchmarking prism{args.field}: "
          f"{args.prisms} prisms x {args.points} points\n")
    for label, bindir in (("CPU", os.path.join(ROOT, "bin")),
                          ("CUDA", os.path.join(ROOT, "bin-cuda"))):
        prog = os.path.join(bindir, "prism" + args.field)
        if not os.path.isfile(prog):
            print(f"{label:5s}: {prog} not found (run: make "
                  f"{'build-cpu' if label == 'CPU' else 'build-cuda'}) — skipped")
            continue
        dt, err = time_run(prog, model, points)
        if dt is None:
            print(f"{label:5s}: FAILED: {err}")
        else:
            per_point = dt/args.points*1e3
            print(f"{label:5s}: {dt:.3f} s total, {per_point:.4f} ms per point "
                  f"({args.prisms*args.points/dt/1e6:.1f} M prism-point evals/s)")

    os.unlink(model)
    os.unlink(points)
    os.rmdir(tmpdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
