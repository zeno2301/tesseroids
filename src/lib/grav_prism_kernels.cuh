/*
CUDA __global__ kernels for the prism gravity fields.

This file is only compiled by nvcc (it is included from grav_prism_cuda.cu).
Each kernel computes one field/component for ALL observation points: one GPU
thread per point, sequential loop over all prisms (same summation order as
the CPU version). The math lives in grav_prism_cuda.h.
*/

#ifndef _TESSEROIDS_GRAV_PRISM_KERNELS_CUH_
#define _TESSEROIDS_GRAV_PRISM_KERNELS_CUH_

#include "grav_prism_cuda.h"

/* Number of threads per block for the kernels */
#define PRISM_CUDA_THREADS 256

/* Define one __global__ kernel per field. */
#define DEF_PRISM_KERNEL(KNAME, FUNC)                                       \
__global__ void KNAME(const PRISM * __restrict__ model, int nprisms,       \
                      const double * __restrict__ points, int npoints,      \
                      double * __restrict__ results)                        \
{                                                                           \
    int p = blockIdx.x*blockDim.x + threadIdx.x;                            \
    if(p >= npoints)                                                        \
    {                                                                       \
        return;                                                             \
    }                                                                       \
    double xp = points[3*p + 0];                                            \
    double yp = points[3*p + 1];                                            \
    double zp = points[3*p + 2];                                            \
    double res = 0.0;                                                       \
    for(int i = 0; i < nprisms; i++)                                        \
    {                                                                       \
        res += FUNC(model + i, xp, yp, zp);                                \
    }                                                                       \
    results[p] = res;                                                       \
}

DEF_PRISM_KERNEL(prism_pot_kernel, prism_pot_dev)
DEF_PRISM_KERNEL(prism_gx_kernel,  prism_gx_dev)
DEF_PRISM_KERNEL(prism_gy_kernel,  prism_gy_dev)
DEF_PRISM_KERNEL(prism_gz_kernel,  prism_gz_dev)
DEF_PRISM_KERNEL(prism_gxx_kernel, prism_gxx_dev)
DEF_PRISM_KERNEL(prism_gxy_kernel, prism_gxy_dev)
DEF_PRISM_KERNEL(prism_gxz_kernel, prism_gxz_dev)
DEF_PRISM_KERNEL(prism_gyy_kernel, prism_gyy_dev)
DEF_PRISM_KERNEL(prism_gyz_kernel, prism_gyz_dev)
DEF_PRISM_KERNEL(prism_gzz_kernel, prism_gzz_dev)

#endif
