/*
CUDA host API for calculating the gravitational potential and its first and
second derivatives of a rectangular prism model on an NVIDIA GPU.

The __global__ kernels live in grav_prism_kernels.cuh; the math they use lives
in grav_prism_cuda.h (shared with the CPU validation harness).
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "grav_prism_kernels.cuh"

/* Copy a CUDA error into errbuf and return the error code */
static int cuda_fail(cudaError_t err, const char *what, char *errbuf,
                     int errbufsize)
{
    if(errbuf != NULL && errbufsize > 0)
    {
        snprintf(errbuf, errbufsize, "%s failed: %s", what,
                 cudaGetErrorString(err));
    }
    return (int)err;
}

#ifdef __cplusplus
extern "C" {
#endif

/* Get the name of a CUDA device. Returns 0 on success. */
int cuda_prism_device_name(int device, char *namebuf, int bufsize)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if(err != cudaSuccess)
    {
        if(namebuf != NULL && bufsize > 0)
        {
            snprintf(namebuf, bufsize, "unknown (%s)",
                     cudaGetErrorString(err));
        }
        return (int)err;
    }
    if(namebuf != NULL && bufsize > 0)
    {
        snprintf(namebuf, bufsize, "%s", prop.name);
    }
    return 0;
}

/* Number of available CUDA devices. Returns -1 if the runtime is unusable. */
int cuda_prism_device_count(void)
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if(err != cudaSuccess)
    {
        return -1;
    }
    return count;
}

/*
Compute a gravity field component of a prism model on a set of points using
the GPU.

@param model      array of prisms (host memory)
@param nprisms    number of prisms in model
@param field      one of PRISM_FIELD_* (see grav_prism_cuda.h)
@param points     array of 3*npoints doubles with the x, y, z coordinates of
                  each observation point (host memory)
@param npoints    number of observation points
@param results    array of npoints doubles to receive the results (host memory)
@param errbuf     buffer for an error message (may be NULL)
@param errbufsize size of errbuf

@return 0 on success, non-zero CUDA error code otherwise.
*/
int cuda_prism_compute(const PRISM *model, int nprisms, int field,
                       const double *points, int npoints, double *results,
                       char *errbuf, int errbufsize)
{
    PRISM *d_model = NULL;
    double *d_points = NULL, *d_results = NULL;
    int blocks, rc = 0;
    size_t modelbytes, pointsbytes, resultsbytes;
    cudaError_t err;

    if(errbuf != NULL && errbufsize > 0)
    {
        errbuf[0] = '\0';
    }
    if(field < 0 || field >= PRISM_FIELD_NFIELDS)
    {
        if(errbuf != NULL && errbufsize > 0)
        {
            snprintf(errbuf, errbufsize, "invalid field id %d", field);
        }
        return -1;
    }
    if(npoints <= 0)
    {
        /* Nothing to do (matches CPU behavior of printing nothing) */
        return 0;
    }
    if(nprisms <= 0 || model == NULL || points == NULL || results == NULL)
    {
        if(errbuf != NULL && errbufsize > 0)
        {
            snprintf(errbuf, errbufsize, "invalid model or points array");
        }
        return -1;
    }

    modelbytes = (size_t)nprisms*sizeof(PRISM);
    pointsbytes = (size_t)npoints*3*sizeof(double);
    resultsbytes = (size_t)npoints*sizeof(double);

    err = cudaMalloc((void **)&d_model, modelbytes);
    if(err != cudaSuccess)
    {
        return cuda_fail(err, "cudaMalloc(model)", errbuf, errbufsize);
    }
    err = cudaMalloc((void **)&d_points, pointsbytes);
    if(err != cudaSuccess)
    {
        rc = cuda_fail(err, "cudaMalloc(points)", errbuf, errbufsize);
        goto cleanup;
    }
    err = cudaMalloc((void **)&d_results, resultsbytes);
    if(err != cudaSuccess)
    {
        rc = cuda_fail(err, "cudaMalloc(results)", errbuf, errbufsize);
        goto cleanup;
    }

    err = cudaMemcpy(d_model, model, modelbytes, cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        rc = cuda_fail(err, "cudaMemcpy(model)", errbuf, errbufsize);
        goto cleanup;
    }
    err = cudaMemcpy(d_points, points, pointsbytes, cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        rc = cuda_fail(err, "cudaMemcpy(points)", errbuf, errbufsize);
        goto cleanup;
    }

    blocks = (npoints + PRISM_CUDA_THREADS - 1)/PRISM_CUDA_THREADS;

    switch(field)
    {
        case PRISM_FIELD_POT:
            prism_pot_kernel<<<blocks, PRISM_CUDA_THREADS>>>(d_model, nprisms,
                d_points, npoints, d_results);
            break;
        case PRISM_FIELD_GX:
            prism_gx_kernel<<<blocks, PRISM_CUDA_THREADS>>>(d_model, nprisms,
                d_points, npoints, d_results);
            break;
        case PRISM_FIELD_GY:
            prism_gy_kernel<<<blocks, PRISM_CUDA_THREADS>>>(d_model, nprisms,
                d_points, npoints, d_results);
            break;
        case PRISM_FIELD_GZ:
            prism_gz_kernel<<<blocks, PRISM_CUDA_THREADS>>>(d_model, nprisms,
                d_points, npoints, d_results);
            break;
        case PRISM_FIELD_GXX:
            prism_gxx_kernel<<<blocks, PRISM_CUDA_THREADS>>>(d_model, nprisms,
                d_points, npoints, d_results);
            break;
        case PRISM_FIELD_GXY:
            prism_gxy_kernel<<<blocks, PRISM_CUDA_THREADS>>>(d_model, nprisms,
                d_points, npoints, d_results);
            break;
        case PRISM_FIELD_GXZ:
            prism_gxz_kernel<<<blocks, PRISM_CUDA_THREADS>>>(d_model, nprisms,
                d_points, npoints, d_results);
            break;
        case PRISM_FIELD_GYY:
            prism_gyy_kernel<<<blocks, PRISM_CUDA_THREADS>>>(d_model, nprisms,
                d_points, npoints, d_results);
            break;
        case PRISM_FIELD_GYZ:
            prism_gyz_kernel<<<blocks, PRISM_CUDA_THREADS>>>(d_model, nprisms,
                d_points, npoints, d_results);
            break;
        case PRISM_FIELD_GZZ:
            prism_gzz_kernel<<<blocks, PRISM_CUDA_THREADS>>>(d_model, nprisms,
                d_points, npoints, d_results);
            break;
    }

    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        rc = cuda_fail(err, "kernel launch", errbuf, errbufsize);
        goto cleanup;
    }
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess)
    {
        rc = cuda_fail(err, "kernel execution", errbuf, errbufsize);
        goto cleanup;
    }

    err = cudaMemcpy(results, d_results, resultsbytes,
                     cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        rc = cuda_fail(err, "cudaMemcpy(results)", errbuf, errbufsize);
        goto cleanup;
    }

cleanup:
    if(d_model != NULL)
    {
        cudaFree(d_model);
    }
    if(d_points != NULL)
    {
        cudaFree(d_points);
    }
    if(d_results != NULL)
    {
        cudaFree(d_results);
    }
    return rc;
}

#ifdef __cplusplus
}
#endif
