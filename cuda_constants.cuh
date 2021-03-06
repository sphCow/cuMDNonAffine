#ifndef CUDA_CONSTANTS
#define CUDA_CONSTANTS

#include "precision.cuh"

__constant__ long dev_N;
__constant__ double dev_rc;
__constant__ double dev_rc2;
__constant__ double3 dev_l;
__constant__ double dev_dt;

__constant__ double3 dev_lcell;
__constant__ uint3 dev_ncell;

#endif
