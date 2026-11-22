/*
CUDA program to calculate gy of a rectangular prism model on a set of points.
Same command line interface and input/output formats as the CPU prismgy.
*/

#include "prismg_main_cuda.h"

/** Main */
int main(int argc, char **argv)
{
    return run_prismg_main_cuda(argc, argv, "prismgy", PRISM_FIELD_GY);
}
