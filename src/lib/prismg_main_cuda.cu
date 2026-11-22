/*
Generic main function for the CUDA versions of the prismg* programs.

Behaves exactly like run_prismg_main() in prismg_main.c (same arguments, same
stdin/stdout formats), but reads all computation points from stdin first and
then calculates all of them in a single batch on the GPU.
*/

/* For strdup() and clock_gettime() when compiled as strict C (nvcc -x c) */
#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "logger.h"
#include "version.h"
#include "geometry.h"
#include "parsers.h"
#include "grav_prism_cuda.h"
#include "prismg_main_cuda.h"

/* Line types for buffered stdin */
#define LINE_ECHO 0  /* comment/blank line: print verbatim */
#define LINE_POINT 1 /* valid computation point */
#define LINE_BAD 2   /* invalid line: skip with a warning */

static char global_progname[100];

/* Print the help message (identical to the CPU version) */
void print_help_cuda()
{
    printf("Usage: %s MODELFILE [OPTIONS]\n\n", global_progname);
    if(strcmp(global_progname + 5, "pot") == 0)
    {
        printf("Calculate the potential due to a rectangular prism model on\n");
    }
    else
    {
        printf("Calculate the %s component due to a rectangular prism model on\n",
                global_progname + 5);
    }
    printf("specified observation points using Cartesian coordinates.\n\n");
    printf("All input units are SI! Output is SI, mGal or Eotvos.\n\n");
    printf("Coordinate system:\n");
    printf("  The coordinate system for the prism is x->North, y->East\n");
    printf("  and z->Down\n\n");
    printf("Input:\n");
    printf("  Computation points passed through standard input (stdin).\n");
    printf("  Reads 3 or more values per line and inteprets the first 3 as:\n");
    printf("    Easting(y) Northing(x) height \n");
    printf("  (the coordinates of a computation point in meters).\n");
    printf("  Other values in the line are ignored.\n");
    printf("  Lines that start with # are ignored as comments.\n");
    printf("  Lines should be no longer than 10000 (ten thousand) characters.");
    printf("  \n\n");
    printf("Output:\n");
    printf("  Printed to standard output (stdout) in the form:\n");
    printf("    y x height ... result\n");
    printf("  ... represents any values that were read from input and\n");
    printf("  ignored. In other words, the result is appended to the last\n");
    printf("  column of the input. Use this to pipe prism* programs\n");
    printf("  together.\n\n");
    printf("  Comments about the provenance of the data are inserted into\n");
    printf("  the top of the output\n\n");
    printf("MODELFILE: File containing the prism model\n");
    printf("  * Each prism is specified by the values of its borders\n");
    printf("    and density\n");
    printf("  * The file should contain one prism per line\n");
    printf("  * If a line starts with # it will be considered a comment and\n");
    printf("    will be ignored.\n");
    printf("  * Each line should have the following column format:\n");
    printf("      X1 X2 Y1 Y2 Z1 Z2 Density\n\n");
    printf("Options:\n");
    printf("  -h           Print instructions.\n");
    printf("  --version    Print version and license information.\n");
    printf("  -v           Enable verbose printing to stderr.\n");
    printf("  -lFILENAME   Print log messages to file FILENAME.\n");
    printf("  Computation is performed on an NVIDIA CUDA GPU.\n");
    print_copyright();
}

/* Grow the line buffers if full. Returns 0 on success, 1 on failure. */
static int grow_line_buffers(char ***lines, int **linetype, int **linepoint,
                             int *cap)
{
    char **nl;
    int *nt, *np;
    *cap *= 2;
    nl = (char **)realloc(*lines, *cap*sizeof(char *));
    nt = (int *)realloc(*linetype, *cap*sizeof(int));
    np = (int *)realloc(*linepoint, *cap*sizeof(int));
    if(nl == NULL || nt == NULL || np == NULL)
    {
        return 1;
    }
    *lines = nl;
    *linetype = nt;
    *linepoint = np;
    return 0;
}

/* Run the main for a generic CUDA prismg* program */
int run_prismg_main_cuda(int argc, char **argv, const char *progname,
                         int field)
{
    BASIC_ARGS args;
    PRISM *model;
    int modelsize, rc, line, error_exit = 0, bad_input = 0;
    int i, nlines = 0, linescap = 1024, npoints = 0, pointscap = 1024;
    char buff[10000];
    char **lines = NULL;
    int *linetype = NULL;
    int *linepoint = NULL; /* index into points for LINE_POINT lines */
    double *points = NULL; /* 3 doubles (x, y, z) per point */
    double *results = NULL;
    double y, x, height;
    FILE *logfile = NULL, *modelfile = NULL;
    time_t rawtime;
    struct tm * timeinfo;
    struct timespec ts_start, ts_end;
    char errbuf[512];
    char devname[256];

    log_init(LOG_INFO);
    strcpy(global_progname, progname);
    rc = parse_basic_args(argc, argv, progname, &args, &print_help_cuda);
    if(rc == 3)
    {
        log_error("%s: missing input file", progname);
        log_warning("Terminating due to bad input");
        log_warning("Try '%s -h' for instructions", progname);
        return 1;
    }
    if(rc == 2)
    {
        return 0;
    }
    if(rc == 1)
    {
        log_warning("Terminating due to bad input");
        log_warning("Try '%s -h' for instructions", progname);
        return 1;
    }
    /* Set the appropriate logging level and log to file if necessary */
    if(!args.verbose)
    {
        log_init(LOG_WARNING);
    }
    if(args.logtofile)
    {
        logfile = fopen(args.logfname, "w");
        if(logfile == NULL)
        {
            log_error("unable to create log file %s", args.logfname);
            log_warning("Terminating due to bad input");
            log_warning("Try '%s -h' for instructions", progname);
            return 1;
        }
        log_tofile(logfile, LOG_INFO);
    }

    /* Check that a CUDA device is available */
    rc = cuda_prism_device_count();
    if(rc <= 0)
    {
        log_error("no NVIDIA CUDA GPU found (cudaGetDeviceCount returned %d)",
                  rc);
        log_warning("Terminating: this program requires a CUDA-capable GPU");
        if(args.logtofile)
            fclose(logfile);
        return 1;
    }
    cuda_prism_device_name(0, devname, sizeof(devname));

    /* Print standard verbose */
    log_info("%s (Tesseroids project, CUDA version) %s", progname,
             tesseroids_version);
    log_info("Using CUDA device 0: %s", devname);
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    log_info("(local time) %s", asctime(timeinfo));

    /* Read the model file */
    log_info("Reading prism model from file %s", args.inputfname);
    modelfile = fopen(args.inputfname, "r");
    if(modelfile == NULL)
    {
        log_error("failed to open model file %s", args.inputfname);
        log_warning("Terminating due to bad input");
        log_warning("Try '%s -h' for instructions", progname);
        if(args.logtofile)
            fclose(logfile);
        return 1;
    }
    model = read_prism_model(modelfile, 0, &modelsize);
    fclose(modelfile);
    if(modelsize == 0)
    {
        log_error("prism file %s is empty", args.inputfname);
        log_warning("Terminating due to bad input");
        log_warning("Try '%s -h' for instructions", progname);
        if(args.logtofile)
            fclose(logfile);
        return 1;
    }
    if(model == NULL)
    {
        log_error("failed to read model from file %s", args.inputfname);
        log_warning("Terminating due to bad input");
        log_warning("Try '%s -h' for instructions", progname);
        if(args.logtofile)
            fclose(logfile);
        return 1;
    }
    log_info("Total of %d prism(s) read", modelsize);

    /* Print a header on the output with provenance information */
    if(strcmp(progname + 5, "pot") == 0)
    {
        printf("# Potential calculated with %s %s:\n", progname,
               tesseroids_version);
    }
    else
    {
        printf("# %s component calculated with %s %s:\n", progname+5, progname,
               tesseroids_version);
    }
    printf("#   local time: %s", asctime(timeinfo));
    printf("#   model file: %s (%d prisms)\n", args.inputfname, modelsize);

    /* Read all computation points from stdin into memory */
    lines = (char **)malloc(linescap*sizeof(char *));
    linetype = (int *)malloc(linescap*sizeof(int));
    linepoint = (int *)malloc(linescap*sizeof(int));
    points = (double *)malloc(pointscap*3*sizeof(double));
    if(lines == NULL || linetype == NULL || linepoint == NULL || points == NULL)
    {
        log_error("problem allocating memory to buffer input points");
        free(lines); free(linetype); free(linepoint); free(points);
        free(model);
        if(args.logtofile)
            fclose(logfile);
        return 1;
    }
    for(line = 1; !feof(stdin); line++)
    {
        if(fgets(buff, 10000, stdin) == NULL)
        {
            if(ferror(stdin))
            {
                log_error("problem encountered reading line %d", line);
                error_exit = 1;
            }
            break;
        }
        if(nlines == linescap && grow_line_buffers(&lines, &linetype,
                                                   &linepoint, &linescap))
        {
            log_error("problem expanding memory for input lines");
            error_exit = 1;
            break;
        }
        /* Check for comments and blank lines */
        if(buff[0] == '#' || buff[0] == '\r' || buff[0] == '\n')
        {
            lines[nlines] = strdup(buff);
            linetype[nlines] = LINE_ECHO;
            linepoint[nlines] = -1;
            nlines++;
            continue;
        }
        if(sscanf(buff, "%lf %lf %lf", &y, &x, &height) != 3)
        {
            log_warning("bad/invalid computation point at line %d", line);
            log_warning("skipping this line and continuing");
            bad_input++;
            lines[nlines] = strdup(buff);
            linetype[nlines] = LINE_BAD;
            linepoint[nlines] = -1;
            nlines++;
            continue;
        }
        if(npoints == pointscap)
        {
            double *tmp;
            pointscap *= 2;
            tmp = (double *)realloc(points, pointscap*3*sizeof(double));
            if(tmp == NULL)
            {
                log_error("problem expanding memory for input points");
                error_exit = 1;
                break;
            }
            points = tmp;
        }
        /* Store x, y, z = -height (z axis points downwards) */
        points[3*npoints + 0] = x;
        points[3*npoints + 1] = y;
        points[3*npoints + 2] = -height;
        strstrip(buff);
        lines[nlines] = strdup(buff);
        linetype[nlines] = LINE_POINT;
        linepoint[nlines] = npoints;
        nlines++;
        npoints++;
    }
    if(bad_input)
    {
        log_warning("Encountered %d bad computation points which were skipped",
                    bad_input);
    }

    /* Calculate all points on the GPU in one batch */
    log_info("Calculating on %d point(s) on CUDA device 0: %s (this may take "
             "a while)...", npoints, devname);
    if(npoints > 0)
    {
        results = (double *)malloc(npoints*sizeof(double));
        if(results == NULL)
        {
            log_error("problem allocating memory for results");
            error_exit = 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    if(!error_exit && npoints > 0)
    {
        rc = cuda_prism_compute(model, modelsize, field, points, npoints,
                                results, errbuf, sizeof(errbuf));
        if(rc != 0)
        {
            log_error("CUDA computation failed: %s", errbuf);
            log_warning("Terminating due to error in computation");
            error_exit = 1;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    /* Print the results in the original input order */
    if(!error_exit)
    {
        for(i = 0; i < nlines; i++)
        {
            if(linetype[i] == LINE_ECHO)
            {
                printf("%s", lines[i]);
            }
            else if(linetype[i] == LINE_POINT)
            {
                printf("%s %.15g\n", lines[i], results[linepoint[i]]);
            }
            /* LINE_BAD lines are skipped, like in the CPU version */
        }
        log_info("Calculated on %d points in %.5g seconds", npoints,
                 (double)(ts_end.tv_sec - ts_start.tv_sec)
                 + 1e-9*(double)(ts_end.tv_nsec - ts_start.tv_nsec));
    }
    if(error_exit)
    {
        log_warning("Terminating due to error in input or computation");
        log_warning("Try '%s -h' for instructions", progname);
    }

    /* Clean up */
    for(i = 0; i < nlines; i++)
    {
        free(lines[i]);
    }
    free(lines);
    free(linetype);
    free(linepoint);
    free(points);
    free(results);
    free(model);
    log_info("Done");
    if(args.logtofile)
        fclose(logfile);
    return error_exit ? 1 : 0;
}
