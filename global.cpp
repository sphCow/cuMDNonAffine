#include "global.h"

double4* host_r = 0;
long  *host_pid = 0;
long  *host_cid = 0;
double4 *host_v = 0;
double4 *host_f = 0;

long  *dev_pid = 0;
long  *dev_cid = 0;
double4 *dev_r = 0;
double4 *dev_v = 0;
double4 *dev_f = 0;

long *dev_pid2gtid = 0;
long *host_pid2gtid = 0;

long nblocks = -1;
long nthreads = -1;
unsigned long tsteps_max = -1;

long N = -1;
double dt = -1;

FileIO *fileIO = new FileIO;
unsigned long tstep = 0;

int write_restart_data_interval = -1;
int restart_data_count = -1;
//*****************************************************//
// array of thermo results, partially reduced over threads

double* dev_pe_reduced = 0;
double4* dev_virial_tensor_pp = 0;
double4* dev_virial_tensor_reduced = 0;
double* dev_virial_pressure_reduced = 0;
double4* host_virial_tensor_pp = 0;

double *dev_v2_reduced = 0; 
double4 *dev_v_tensor_reduced = 0; 

/*********************** functions ************************/
void set_accumulators_zero() {
	cudaMemset(dev_f, 0, N*sizeof(double4));
	cudaMemset(dev_virial_tensor_pp, 0, N*sizeof(double4));
	
	cudaMemset(dev_pe_reduced, 0, nblocks*sizeof(double));	
	cudaMemset(dev_virial_tensor_reduced, 0, nblocks*sizeof(double4));
	cudaMemset(dev_virial_pressure_reduced, 0, nblocks*sizeof(double));

}
