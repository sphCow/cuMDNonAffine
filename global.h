#ifndef __GLOBAL__
#define __GLOBAL__

#include <iostream>
#include <string>
#include <fstream>

#include "vector_types.h"
#include "FileIO.h"

#include "cuda.h"
#include "cuda_runtime.h"

// cpu_arrays
extern double4* host_r;
extern long  *host_pid;
extern long  *host_cid;
extern double4 *host_v;
extern double4 *host_f;

//gpu arrays
extern long *dev_pid2gtid;
extern long *host_pid2gtid; //DEBUG

extern long *dev_pid;
extern long  *dev_cid;
extern double4 *dev_r;
extern double4 *dev_v;
extern double4 *dev_f;

extern long N;

// array of thermo results, partially reduced over threads
extern double* dev_pe_reduced;        // pe, [reduced over blocks]
extern double4* dev_virial_tensor_pp;     // 4 component virial tensor [per particle]
extern double4* dev_virial_tensor_reduced; // 4 compoent virial tensor [reduced over blocks]
extern double* dev_virial_pressure_reduced;     // sigma = sigma_xx + sigma_yy [reduced over blocks]
extern double *dev_v2_reduced; 
extern double4 *dev_v_tensor_reduced;


//virial tensor
extern double4* host_virial_tensor_pp;

enum UseDevice {CPU, GPU};

//fileIO
extern FileIO *fileIO;

//tstep
extern unsigned long tsteps_max;
extern 	unsigned long tstep;
extern double dt;

//restart_data
extern int write_restart_data_interval;
extern int restart_data_count;

//kernel config
extern long nblocks;
extern long nthreads;

// Array modifiers
extern void set_accumulators_zero();

#endif
