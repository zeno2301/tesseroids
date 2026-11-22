# CUDA version of the prism* programs

This directory contains a CUDA port of the rectangular-prism forward modelling
programs (`prismpot`, `prismgx`, `prismgy`, `prismgz`, `prismgxx`, `prismgxy`,
`prismgxz`, `prismgyy`, `prismgyz`, `prismgzz`).

The CUDA programs have **exactly the same command line interface and input/output
formats** as the CPU versions:

    bin-cuda/prismgz MODELFILE [OPTIONS] < points.txt > result.txt

* `MODELFILE` — one prism per line: `X1 X2 Y1 Y2 Z1 Z2 Density` (SI units,
  lines starting with `#` are comments).
* Computation points are read from stdin, 3 or more values per line, first 3
  interpreted as `Easting(y) Northing(x) height` (meters).
* Output on stdout: each input line with the result appended in the last
  column (`%.15g`). Comment lines are echoed verbatim. Units: potential in SI,
  gx/gy/gz in mGal, gradient tensor components in Eotvos — identical to the CPU
  programs.
* Options: `-h`, `--version`, `-v`, `-lFILENAME` — same as the CPU programs.

The only difference is that the computation is done on an NVIDIA GPU.

## How it works

* `src/lib/grav_prism_cuda.h` — device math (Nagy et al. 2000 formulas),
  shared between nvcc (`__device__ __host__`) and a plain C compiler
  (`static inline`), which is what allows the exact device code to be unit
  tested on the CPU (see `test/test_grav_prism_cuda.c`).
* `src/lib/grav_prism_kernels.cuh` — one `__global__` kernel per field.
  Each GPU thread handles one observation point and loops over all prisms
  (same summation order as the CPU version, so results match bit-for-bit up to
  libm differences in `atan2`/`log`/`sqrt`).
* `src/lib/grav_prism_cuda.cu` — host API: uploads the model and the points,
  launches the kernel, downloads the results.
* `src/lib/prismg_main_cuda.cu` — CLI runner, identical behaviour to
  `prismg_main.c` (buffers stdin, computes all points in one GPU batch, prints
  results in the original input order).
* `src/cuda/prism*.c` — the 10 program entry points.

## Building

You need the CUDA toolkit (`nvcc`) and, to *run*, an NVIDIA GPU.

    make build-cuda          # builds bin-cuda/prism* with nvcc
    make build-cpu           # builds the original CPU bin/prism* with gcc
    make test-cpu            # builds and runs the unit tests with gcc

Build settings can be overridden, e.g.:

    make build-cuda NVCCARCH="-arch=native" CUDA_PATH=/opt/cuda

By default the binaries are compiled for `compute_60` PTX (JIT-compilable on
any GPU sm_60 and newer) plus native SASS for sm_75.

## Checking that CUDA and CPU results agree

On a machine with a GPU:

    make build-cpu build-cuda
    python3 test/compare_cpu_cuda.py            # uses generated test data
    # or with your own files:
    python3 test/compare_cpu_cuda.py model.txt points.txt

The script runs every `prism*` program in both `bin/` and `bin-cuda/` and
compares the outputs numerically.

## Benchmarking

    python3 bench/bench_prisms.py --prisms 100000 --points 2601 --field gz

This generates a random model and a grid of points, then times both the CPU
and the CUDA `prismgz` (CPU numbers only if `bin/` is built, CUDA numbers only
if a GPU is present).

## Notes

* All points from stdin are computed in a single GPU batch, so the whole input
  is buffered in memory first. Output content is identical to the CPU version.
* If no CUDA device is found, the programs print an error to stderr and exit
  with status 1.
* The tesseroid programs (`tess*`) are not ported yet.
