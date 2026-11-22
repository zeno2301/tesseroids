/*
Generic main function for the CUDA versions of the prismg* programs.

The command line interface, input and output formats are identical to the
CPU versions (see prismg_main.h). The computation is done on an NVIDIA GPU.
*/

#ifndef _TESSEROIDS_PRISMG_MAIN_CUDA_H_
#define _TESSEROIDS_PRISMG_MAIN_CUDA_H_

#include "grav_prism_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Print the help message */
extern void print_help_cuda();

/** Run the main for a generic CUDA prismg* program

@param argc number of command line arguments
@param argv command line arguments
@param progname name of the specific program (e.g. "prismgz")
@param field one of PRISM_FIELD_* identifying which component to calculate

@return 0 if all went well, 1 if failed.
*/
extern int run_prismg_main_cuda(int argc, char **argv, const char *progname,
                                int field);

#ifdef __cplusplus
}
#endif

#endif
