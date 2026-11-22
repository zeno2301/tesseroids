/*
CUDA-friendly implementations of the gravitational potential and its first and
second derivatives for a rectangular prism, using the formulas in
Nagy et al. (2000).

This header is shared between the CUDA kernels (compiled by nvcc, where
CUDA_DEV expands to __device__ __host__) and plain C test harnesses (compiled
by gcc, where CUDA_DEV expands to static inline). This allows the exact same
device code to be validated numerically on the CPU.

The formulas are identical to grav_prism.c. The only difference is that
pow(-1, i+j+k) is replaced by an explicit sign, which is bit-identical and
much cheaper on the GPU.

The coordinate system used is that of the article, ie:

x -> North  y -> East  z -> Down

References
----------

* Nagy, D., Papp, G., Benedek, J. (2000): The gravitational potential and its
  derivatives for the prism. Journal of Geodesy, 74, 552-560.
*/

#ifndef _TESSEROIDS_GRAV_PRISM_CUDA_H_
#define _TESSEROIDS_GRAV_PRISM_CUDA_H_

#include <math.h>
#include "geometry.h"

/* Physical constants as compile-time literals.
   Device code cannot access the extern host globals from constants.c, so the
   kernels use these literals instead. The values are bit-identical to the
   ones in constants.c (same decimal literals rounded to double). */
#define CUDA_PI       3.14159265358979323846264338327950288
#define CUDA_G        6.673e-11
#define CUDA_SI2MGAL  100000.0
#define CUDA_SI2EOTVOS 1000000000.0

/* When compiled by nvcc, functions are usable on both device and host.
   When compiled by a plain C compiler (e.g. gcc, for CPU validation of the
   device code), they become static inline host functions. */
#ifdef __CUDACC__
#define CUDA_DEV __device__ __host__
#else
#define CUDA_DEV static inline
#endif

/* Identifiers for each field/component that can be computed on the GPU.
   The order matches the prism* program names. */
enum PRISM_FIELD {
    PRISM_FIELD_POT = 0,
    PRISM_FIELD_GX,
    PRISM_FIELD_GY,
    PRISM_FIELD_GZ,
    PRISM_FIELD_GXX,
    PRISM_FIELD_GXY,
    PRISM_FIELD_GXZ,
    PRISM_FIELD_GYY,
    PRISM_FIELD_GYZ,
    PRISM_FIELD_GZZ,
    PRISM_FIELD_NFIELDS
};

/* Sign of (-1)^n without calling pow() (bit-identical, cheap on GPU) */
CUDA_DEV double cuda_sign(int n)
{
    return (n & 1) ? -1.0 : 1.0;
}

CUDA_DEV double cuda_safe_atan2(double y, double x)
{
    if(y == 0)
    {
        return 0;
    }
    if((y > 0) && (x < 0))
    {
        return atan2(y, x) - CUDA_PI;
    }
    if((y < 0) && (x < 0))
    {
        return atan2(y, x) + CUDA_PI;
    }
    return atan2(y, x);
}

CUDA_DEV double cuda_safe_log(double x)
{
    if(x == 0)
    {
        return 0;
    }
    else
    {
        return log(x);
    }
}

/* Calculates the potential caused by a prism. */
CUDA_DEV double prism_pot_dev(const PRISM *prism, double xp, double yp,
                              double zp)
{
    double x[2], y[2], z[2], kernel, res, r;
    int i, j, k;

    /* First thing to do is make P the origin of the coordinate system */
    x[0] = prism->x2 - xp;
    x[1] = prism->x1 - xp;
    y[0] = prism->y2 - yp;
    y[1] = prism->y1 - yp;
    z[0] = prism->z2 - zp;
    z[1] = prism->z1 - zp;

    res = 0;

    /* Evaluate the integration limits */
    for(k = 0; k <= 1; k++)
    {
        for(j = 0; j <= 1; j++)
        {
            for(i = 0; i <= 1; i++)
            {
                r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);
                kernel = (x[i]*y[j]*cuda_safe_log(z[k] + r)
                          + y[j]*z[k]*cuda_safe_log(x[i] + r)
                          + x[i]*z[k]*cuda_safe_log(y[j] + r)
                          - 0.5*x[i]*x[i]*cuda_safe_atan2(z[k]*y[j], x[i]*r)
                          - 0.5*y[j]*y[j]*cuda_safe_atan2(z[k]*x[i], y[j]*r)
                          - 0.5*z[k]*z[k]*cuda_safe_atan2(x[i]*y[j], z[k]*r));
                res += cuda_sign(i + j + k)*kernel;
            }
        }
    }

    /* Now all that is left is to multiply res by the gravitational constant
       and density */
    res *= CUDA_G*prism->density;

    return res;
}

/* Calculates the x component of gravitational attraction caused by a prism. */
CUDA_DEV double prism_gx_dev(const PRISM *prism, double xp, double yp,
                             double zp)
{
    double x[2], y[2], z[2], kernel, res, r;
    int i, j, k;

    x[0] = prism->x2 - xp;
    x[1] = prism->x1 - xp;
    y[0] = prism->y2 - yp;
    y[1] = prism->y1 - yp;
    z[0] = prism->z2 - zp;
    z[1] = prism->z1 - zp;

    res = 0;

    for(k = 0; k <= 1; k++)
    {
        for(j = 0; j <= 1; j++)
        {
            for(i = 0; i <= 1; i++)
            {
                r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);

                kernel = -(y[j]*cuda_safe_log(z[k] + r)
                           + z[k]*cuda_safe_log(y[j] + r)
                           - x[i]*cuda_safe_atan2(z[k]*y[j], x[i]*r));

                res += cuda_sign(i + j + k)*kernel;
            }
        }
    }

    res *= CUDA_G*CUDA_SI2MGAL*prism->density;

    return res;
}

/* Calculates the y component of gravitational attraction caused by a prism. */
CUDA_DEV double prism_gy_dev(const PRISM *prism, double xp, double yp,
                             double zp)
{
    double x[2], y[2], z[2], kernel, res, r;
    int i, j, k;

    x[0] = prism->x2 - xp;
    x[1] = prism->x1 - xp;
    y[0] = prism->y2 - yp;
    y[1] = prism->y1 - yp;
    z[0] = prism->z2 - zp;
    z[1] = prism->z1 - zp;

    res = 0;

    for(k = 0; k <= 1; k++)
    {
        for(j = 0; j <= 1; j++)
        {
            for(i = 0; i <= 1; i++)
            {
                r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);

                kernel = -(z[k]*cuda_safe_log(x[i] + r)
                           + x[i]*cuda_safe_log(z[k] + r)
                           - y[j]*cuda_safe_atan2(z[k]*x[i], y[j]*r));

                res += cuda_sign(i + j + k)*kernel;
            }
        }
    }

    res *= CUDA_G*CUDA_SI2MGAL*prism->density;

    return res;
}

/* Calculates the z component of gravitational attraction caused by a prism. */
CUDA_DEV double prism_gz_dev(const PRISM *prism, double xp, double yp,
                             double zp)
{
    double x[2], y[2], z[2], kernel, res, r;
    int i, j, k;

    x[0] = prism->x2 - xp;
    x[1] = prism->x1 - xp;
    y[0] = prism->y2 - yp;
    y[1] = prism->y1 - yp;
    z[0] = prism->z2 - zp;
    z[1] = prism->z1 - zp;

    res = 0;

    for(k = 0; k <= 1; k++)
    {
        for(j = 0; j <= 1; j++)
        {
            for(i = 0; i <= 1; i++)
            {
                r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);

                kernel = -(x[i]*cuda_safe_log(y[j] + r)
                           + y[j]*cuda_safe_log(x[i] + r)
                           - z[k]*cuda_safe_atan2(x[i]*y[j], z[k]*r));

                res += cuda_sign(i + j + k)*kernel;
            }
        }
    }

    res *= CUDA_G*CUDA_SI2MGAL*prism->density;

    return res;
}

/* Calculates the gxx gravity gradient tensor component caused by a prism. */
CUDA_DEV double prism_gxx_dev(const PRISM *prism, double xp, double yp,
                              double zp)
{
    double x[2], y[2], z[2], kernel, res, r;
    int i, j, k;

    x[0] = prism->x2 - xp;
    x[1] = prism->x1 - xp;
    y[0] = prism->y2 - yp;
    y[1] = prism->y1 - yp;
    z[0] = prism->z2 - zp;
    z[1] = prism->z1 - zp;

    res = 0;

    for(k = 0; k <= 1; k++)
    {
        for(j = 0; j <= 1; j++)
        {
            for(i = 0; i <= 1; i++)
            {
                r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);
                kernel = -cuda_safe_atan2(z[k]*y[j], x[i]*r);
                res += cuda_sign(i + j + k)*kernel;
            }
        }
    }

    res *= CUDA_G*CUDA_SI2EOTVOS*prism->density;

    return res;
}

/* Calculates the gxy gravity gradient tensor component caused by a prism. */
CUDA_DEV double prism_gxy_dev(const PRISM *prism, double xp, double yp,
                              double zp)
{
    double x[2], y[2], z[2], kernel, res, r, xtmp, ytmp;
    int i, j, k;

    x[0] = prism->x2 - xp;
    x[1] = prism->x1 - xp;
    y[0] = prism->y2 - yp;
    y[1] = prism->y1 - yp;
    z[0] = prism->z2 - zp;
    z[1] = prism->z1 - zp;

    res = 0;

    for(k = 0; k <= 1; k++)
    {
        for(j = 0; j <= 1; j++)
        {
            for(i = 0; i <= 1; i++)
            {
                if(x[i] == 0 && y[j] == 0 && z[k] < 0)
                {
                    xtmp = 0.0001*(prism->x2 - prism->x1);
                    ytmp = 0.0001*(prism->y2 - prism->y1);
                    r = sqrt(xtmp*xtmp + ytmp*ytmp + z[k]*z[k]);
                }
                else
                {
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);
                }
                kernel = cuda_safe_log(z[k] + r);
                res += cuda_sign(i + j + k)*kernel;
            }
        }
    }

    res *= CUDA_G*CUDA_SI2EOTVOS*prism->density;

    return res;
}

/* Calculates the gxz gravity gradient tensor component caused by a prism. */
CUDA_DEV double prism_gxz_dev(const PRISM *prism, double xp, double yp,
                              double zp)
{
    double x[2], y[2], z[2], kernel, res, r, xtmp, ztmp;
    int i, j, k;

    x[0] = prism->x2 - xp;
    x[1] = prism->x1 - xp;
    y[0] = prism->y2 - yp;
    y[1] = prism->y1 - yp;
    z[0] = prism->z2 - zp;
    z[1] = prism->z1 - zp;

    res = 0;

    for(k = 0; k <= 1; k++)
    {
        for(j = 0; j <= 1; j++)
        {
            for(i = 0; i <= 1; i++)
            {
                if(x[i] == 0 && z[k] == 0 && y[j] < 0)
                {
                    xtmp = 0.0001*(prism->x2 - prism->x1);
                    ztmp = 0.0001*(prism->z2 - prism->z1);
                    r = sqrt(xtmp*xtmp + ztmp*ztmp + y[j]*y[j]);
                }
                else
                {
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);
                }
                kernel = cuda_safe_log(y[j] + r);
                res += cuda_sign(i + j + k)*kernel;
            }
        }
    }

    res *= CUDA_G*CUDA_SI2EOTVOS*prism->density;

    return res;
}

/* Calculates the gyy gravity gradient tensor component caused by a prism. */
CUDA_DEV double prism_gyy_dev(const PRISM *prism, double xp, double yp,
                              double zp)
{
    double x[2], y[2], z[2], kernel, res, r;
    int i, j, k;

    x[0] = prism->x2 - xp;
    x[1] = prism->x1 - xp;
    y[0] = prism->y2 - yp;
    y[1] = prism->y1 - yp;
    z[0] = prism->z2 - zp;
    z[1] = prism->z1 - zp;

    res = 0;

    for(k = 0; k <= 1; k++)
    {
        for(j = 0; j <= 1; j++)
        {
            for(i = 0; i <= 1; i++)
            {
                r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);
                kernel = -cuda_safe_atan2(z[k]*x[i], y[j]*r);
                res += cuda_sign(i + j + k)*kernel;
            }
        }
    }

    res *= CUDA_G*CUDA_SI2EOTVOS*prism->density;

    return res;
}

/* Calculates the gyz gravity gradient tensor component caused by a prism. */
CUDA_DEV double prism_gyz_dev(const PRISM *prism, double xp, double yp,
                              double zp)
{
    double x[2], y[2], z[2], kernel, res, r, ytmp, ztmp;
    int i, j, k;

    x[0] = prism->x2 - xp;
    x[1] = prism->x1 - xp;
    y[0] = prism->y2 - yp;
    y[1] = prism->y1 - yp;
    z[0] = prism->z2 - zp;
    z[1] = prism->z1 - zp;

    res = 0;

    for(k = 0; k <= 1; k++)
    {
        for(j = 0; j <= 1; j++)
        {
            for(i = 0; i <= 1; i++)
            {
                if(z[k] == 0 && y[j] == 0 && x[i] < 0)
                {
                    ytmp = 0.0001*(prism->y2 - prism->y1);
                    ztmp = 0.0001*(prism->z2 - prism->z1);
                    r = sqrt(ztmp*ztmp + ytmp*ytmp + x[i]*x[i]);
                }
                else
                {
                    r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);
                }
                kernel = cuda_safe_log(x[i] + r);
                res += cuda_sign(i + j + k)*kernel;
            }
        }
    }

    res *= CUDA_G*CUDA_SI2EOTVOS*prism->density;

    return res;
}

/* Calculates the gzz gravity gradient tensor component caused by a prism. */
CUDA_DEV double prism_gzz_dev(const PRISM *prism, double xp, double yp,
                              double zp)
{
    double x[2], y[2], z[2], kernel, res, r;
    int i, j, k;

    x[0] = prism->x2 - xp;
    x[1] = prism->x1 - xp;
    y[0] = prism->y2 - yp;
    y[1] = prism->y1 - yp;
    z[0] = prism->z2 - zp;
    z[1] = prism->z1 - zp;

    res = 0;

    for(k = 0; k <= 1; k++)
    {
        for(j = 0; j <= 1; j++)
        {
            for(i = 0; i <= 1; i++)
            {
                r = sqrt(x[i]*x[i] + y[j]*y[j] + z[k]*z[k]);
                kernel = -cuda_safe_atan2(x[i]*y[j], z[k]*r);
                res += cuda_sign(i + j + k)*kernel;
            }
        }
    }

    res *= CUDA_G*CUDA_SI2EOTVOS*prism->density;

    return res;
}

/* Host-side API implemented in grav_prism_cuda.cu (compiled by nvcc) */
#ifdef __cplusplus
extern "C" {
#endif
/* Number of available CUDA devices. Returns -1 if the runtime is unusable. */
int cuda_prism_device_count(void);
/* Get the name of a CUDA device into namebuf. Returns 0 on success. */
int cuda_prism_device_name(int device, char *namebuf, int bufsize);
/* Compute a field component of the prism model on the GPU (see
   grav_prism_cuda.cu). Returns 0 on success. */
int cuda_prism_compute(const PRISM *model, int nprisms, int field,
                       const double *points, int npoints, double *results,
                       char *errbuf, int errbufsize);
#ifdef __cplusplus
}
#endif

/* Dispatch a field id to the matching device function.
   Only used on the host side to select the kernel; the kernels themselves
   call the device functions directly. */
CUDA_DEV double prism_field_dev(int field, const PRISM *prism, double xp,
                                double yp, double zp)
{
    switch(field)
    {
        case PRISM_FIELD_POT: return prism_pot_dev(prism, xp, yp, zp);
        case PRISM_FIELD_GX:  return prism_gx_dev(prism, xp, yp, zp);
        case PRISM_FIELD_GY:  return prism_gy_dev(prism, xp, yp, zp);
        case PRISM_FIELD_GZ:  return prism_gz_dev(prism, xp, yp, zp);
        case PRISM_FIELD_GXX: return prism_gxx_dev(prism, xp, yp, zp);
        case PRISM_FIELD_GXY: return prism_gxy_dev(prism, xp, yp, zp);
        case PRISM_FIELD_GXZ: return prism_gxz_dev(prism, xp, yp, zp);
        case PRISM_FIELD_GYY: return prism_gyy_dev(prism, xp, yp, zp);
        case PRISM_FIELD_GYZ: return prism_gyz_dev(prism, xp, yp, zp);
        case PRISM_FIELD_GZZ: return prism_gzz_dev(prism, xp, yp, zp);
        default: return 0.0;
    }
}

#endif
