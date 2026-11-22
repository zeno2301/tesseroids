/*
CUDA program to calculate gyy of a rectangular prism model on a set of points.
Same command line interface and input/output formats as the CPU prismgyy.
*/

#include "prismg_main_cuda.h"

/** Main */
int main(int argc, char **argv)
{
    return run_prismg_main_cuda(argc, argv, "prismgyy", PRISM_FIELD_GYY);
}
