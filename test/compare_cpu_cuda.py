#!/usr/bin/env python3
"""
Compare the outputs of the CPU (bin/) and CUDA (bin-cuda/) prism* programs.

Usage:
    python3 test/compare_cpu_cuda.py [MODELFILE POINTSFILE]

Without arguments, a random model (1000 prisms) and a 21x21 grid of points are
generated in a temporary directory. With arguments, the given model and points
files are used.

For every component (pot, gx, gy, gz, gxx, gxy, gxz, gyy, gyz, gzz) both
programs are run and the numeric outputs are compared with a relative
tolerance. Prints PASS/FAIL per component and exits non-zero on any mismatch.
"""
import os
import subprocess
import sys
import tempfile
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN_CPU = os.path.join(ROOT, "bin")
BIN_CUDA = os.path.join(ROOT, "bin-cuda")
FIELDS = ["pot", "gx", "gy", "gz", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz"]
RTOL = 1e-9  # relative tolerance for numeric comparison


def gen_model(path, n=1000, seed=42):
    rng = random.Random(seed)
    with open(path, "w") as f:
        for _ in range(n):
            x1 = rng.uniform(-10000, 10000); dx = rng.uniform(50, 500)
            y1 = rng.uniform(-10000, 10000); dy = rng.uniform(50, 500)
            z1 = rng.uniform(0, 2000);      dz = rng.uniform(50, 500)
            dens = rng.uniform(2000, 3500)
            f.write(f"{x1} {x1+dx} {y1} {y1+dy} {z1} {z1+dz} {dens}\n")


def gen_points(path, nx=21, ny=21, step=100.0, height=2000.0):
    with open(path, "w") as f:
        f.write("# test points grid\n")
        for i in range(nx):
            for j in range(ny):
                f.write(f"{i*step} {j*step} {height}\n")


def run(prog, model, points_file):
    with open(points_file) as fin:
        p = subprocess.run([prog, model], stdin=fin, capture_output=True,
                           text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{prog} exited with {p.returncode}: {p.stderr}")
    vals = []
    for line in p.stdout.splitlines():
        if line.startswith("#"):
            continue
        vals.append(float(line.split()[-1]))
    return vals


def main():
    if len(sys.argv) == 3:
        model, points = sys.argv[1], sys.argv[2]
        tmpdir = None
    else:
        tmpdir = tempfile.mkdtemp(prefix="tesseroids_compare_")
        model = os.path.join(tmpdir, "model.txt")
        points = os.path.join(tmpdir, "points.txt")
        gen_model(model)
        gen_points(points)
        print(f"Generated test model: {model}")
        print(f"Generated test points: {points}")

    failed = 0
    for f in FIELDS:
        cpu_prog = os.path.join(BIN_CPU, "prism" + f)
        cuda_prog = os.path.join(BIN_CUDA, "prism" + f)
        if not (os.path.isfile(cpu_prog) and os.path.isfile(cuda_prog)):
            print(f"prism{f}: SKIP (build bin/ and bin-cuda/ first)")
            failed += 1
            continue
        a = run(cpu_prog, model, points)
        b = run(cuda_prog, model, points)
        if len(a) != len(b):
            print(f"prism{f}: FAIL (different number of output lines)")
            failed += 1
            continue
        worst = 0.0
        ok = True
        for x, y in zip(a, b):
            denom = max(abs(x), abs(y), 1e-30)
            rel = abs(x - y)/denom
            worst = max(worst, rel)
            if rel > RTOL:
                ok = False
        if ok:
            print(f"prism{f}: PASS (max rel diff {worst:.3g} over {len(a)} points)")
        else:
            print(f"prism{f}: FAIL (max rel diff {worst:.3g} > {RTOL:g})")
            failed += 1

    if tmpdir:
        for p in (model, points):
            os.unlink(p)
        os.rmdir(tmpdir)

    if failed:
        print(f"\n{failed} component(s) FAILED or skipped")
        return 1
    print("\nAll components match between CPU and CUDA versions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
