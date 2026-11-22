/*
CUDA program to calculate pot of a rectangular prism model on a set of points.
Same command line interface and input/output formats as the CPU prismpot.
*/

#include "prismg_main_cuda.h"

/** Main */
int main(int argc, char **argv)
{
    return run_prismg_main_cuda(argc, argv, "prismpot", PRISM_FIELD_POT);
}
