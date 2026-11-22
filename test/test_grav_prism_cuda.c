/*
Unit tests for the CUDA device functions in grav_prism_cuda.h.

The header is compiled here as plain C (CUDA_DEV = static inline), so these
tests validate on the CPU that the exact device code used by the CUDA kernels
produces the same results as the reference implementation in grav_prism.c.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/lib/grav_prism.h"
#include "../src/lib/grav_prism_cuda.h"
#include "../src/lib/geometry.h"
#include "../src/lib/constants.h"

/* msg is declared in test_grav_tess.c (included before this file) */

/* Simple deterministic PRNG so failures are reproducible */
static unsigned long rng_state = 123456789UL;
static double frand(double lo, double hi)
{
    rng_state = rng_state*6364136223846793005UL + 1442695040888963407UL;
    double u = (double)((rng_state >> 11) & 0xFFFFFFFFFFFFFUL)
               / (double)0x10000000000000UL;
    return lo + u*(hi - lo);
}

/* Compare a device function against the matching CPU reference function on
   random prisms and points, including points inside the prism */
static char * check_field(const char *name,
                          double (*devf)(const PRISM *, double, double,
                                         double),
                          double (*cpuf)(PRISM, double, double, double),
                          int ntests)
{
    int t;
    for(t = 0; t < ntests; t++)
    {
        PRISM prism;
        double x1, dx, y1, dy, z1, dz, xp, yp, zp;
        double ref, got, tol;
        prism.x1 = x1 = frand(-10000, 10000);
        prism.x2 = x1 + (dx = frand(10, 2000));
        prism.y1 = y1 = frand(-10000, 10000);
        prism.y2 = y1 + (dy = frand(10, 2000));
        prism.z1 = z1 = frand(-5000, 5000);
        prism.z2 = z1 + (dz = frand(10, 2000));
        prism.density = frand(1000, 4000);
        /* Mix of outside points and points inside/near the prism */
        if(t % 3 == 0)
        {
            xp = frand(prism.x1 - 3*dx, prism.x2 + 3*dx);
            yp = frand(prism.y1 - 3*dy, prism.y2 + 3*dy);
            zp = frand(prism.z1 - 3*dz, prism.z2 + 3*dz);
        }
        else if(t % 3 == 1)
        {
            /* exactly on a corner/edge of the prism */
            xp = (t % 2) ? prism.x1 : prism.x2;
            yp = (t % 2) ? prism.y2 : prism.y1;
            zp = (t % 2) ? prism.z1 : prism.z2;
        }
        else
        {
            xp = frand(-20000, 20000);
            yp = frand(-20000, 20000);
            zp = frand(-20000, 0);
        }
        ref = cpuf(prism, xp, yp, zp);
        got = devf(&prism, xp, yp, zp);
        tol = 1e-12*(fabs(ref) + 1e-30);
        sprintf(msg, "%s mismatch: cpu=%.17g dev=%.17g (tol %g) at "
                "point (%g, %g, %g)", name, ref, got, tol, xp, yp, zp);
        mu_assert(fabs(ref - got) <= tol, msg);
    }
    return 0;
}

static char * test_cuda_pot_matches_cpu()
{
    return check_field("pot", prism_pot_dev, prism_pot, 2000);
}
static char * test_cuda_gx_matches_cpu()
{
    return check_field("gx", prism_gx_dev, prism_gx, 2000);
}
static char * test_cuda_gy_matches_cpu()
{
    return check_field("gy", prism_gy_dev, prism_gy, 2000);
}
static char * test_cuda_gz_matches_cpu()
{
    return check_field("gz", prism_gz_dev, prism_gz, 2000);
}
static char * test_cuda_gxx_matches_cpu()
{
    return check_field("gxx", prism_gxx_dev, prism_gxx, 2000);
}
static char * test_cuda_gxy_matches_cpu()
{
    return check_field("gxy", prism_gxy_dev, prism_gxy, 2000);
}
static char * test_cuda_gxz_matches_cpu()
{
    return check_field("gxz", prism_gxz_dev, prism_gxz, 2000);
}
static char * test_cuda_gyy_matches_cpu()
{
    return check_field("gyy", prism_gyy_dev, prism_gyy, 2000);
}
static char * test_cuda_gyz_matches_cpu()
{
    return check_field("gyz", prism_gyz_dev, prism_gyz, 2000);
}
static char * test_cuda_gzz_matches_cpu()
{
    return check_field("gzz", prism_gzz_dev, prism_gzz, 2000);
}

/* The dispatch helper must agree with the individual functions */
static char * test_cuda_dispatch()
{
    PRISM prism = {2670, -1000, 1000, -2000, 2000, 0, 500, 0, 0, 0};
    double xp = 300, yp = -700, zp = -250;
    double expect, got;
    int f;
    for(f = 0; f < PRISM_FIELD_NFIELDS; f++)
    {
        expect = prism_field_dev(f, &prism, xp, yp, zp);
        switch(f)
        {
            case PRISM_FIELD_POT: got = prism_pot_dev(&prism, xp, yp, zp); break;
            case PRISM_FIELD_GX:  got = prism_gx_dev(&prism, xp, yp, zp); break;
            case PRISM_FIELD_GY:  got = prism_gy_dev(&prism, xp, yp, zp); break;
            case PRISM_FIELD_GZ:  got = prism_gz_dev(&prism, xp, yp, zp); break;
            case PRISM_FIELD_GXX: got = prism_gxx_dev(&prism, xp, yp, zp); break;
            case PRISM_FIELD_GXY: got = prism_gxy_dev(&prism, xp, yp, zp); break;
            case PRISM_FIELD_GXZ: got = prism_gxz_dev(&prism, xp, yp, zp); break;
            case PRISM_FIELD_GYY: got = prism_gyy_dev(&prism, xp, yp, zp); break;
            case PRISM_FIELD_GYZ: got = prism_gyz_dev(&prism, xp, yp, zp); break;
            default:              got = prism_gzz_dev(&prism, xp, yp, zp); break;
        }
        sprintf(msg, "dispatch mismatch for field %d: %.17g vs %.17g",
                f, expect, got);
        mu_assert(expect == got, msg);
    }
    return 0;
}

int grav_prism_cuda_run_all()
{
    int failed = 0;
    failed += mu_run_test(test_cuda_pot_matches_cpu,
                "CUDA pot device code matches CPU prism_pot");
    failed += mu_run_test(test_cuda_gx_matches_cpu,
                "CUDA gx device code matches CPU prism_gx");
    failed += mu_run_test(test_cuda_gy_matches_cpu,
                "CUDA gy device code matches CPU prism_gy");
    failed += mu_run_test(test_cuda_gz_matches_cpu,
                "CUDA gz device code matches CPU prism_gz");
    failed += mu_run_test(test_cuda_gxx_matches_cpu,
                "CUDA gxx device code matches CPU prism_gxx");
    failed += mu_run_test(test_cuda_gxy_matches_cpu,
                "CUDA gxy device code matches CPU prism_gxy");
    failed += mu_run_test(test_cuda_gxz_matches_cpu,
                "CUDA gxz device code matches CPU prism_gxz");
    failed += mu_run_test(test_cuda_gyy_matches_cpu,
                "CUDA gyy device code matches CPU prism_gyy");
    failed += mu_run_test(test_cuda_gyz_matches_cpu,
                "CUDA gyz device code matches CPU prism_gyz");
    failed += mu_run_test(test_cuda_gzz_matches_cpu,
                "CUDA gzz device code matches CPU prism_gzz");
    failed += mu_run_test(test_cuda_dispatch,
                "CUDA field dispatch matches individual device functions");
    return failed;
}
