#ifndef __NONBONDED_FORCES_H__
#define __NONBONDED_FORCES_H__

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <cmath>

#include <boost/lexical_cast.hpp>

#include "vector_types.h"
#include "global.h"
#include "cuda_constants.cuh"

#include <cub/cub.cuh>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include "kernels/sum4.h"

extern "C" void map_pid2gtid(long *dev_pid,long *dev_map_pid2gtid,const long &N ,const int &nblocks,const int &nthreads);
extern double GlobalReduce(double *d_in, int N);
extern double4 GlobalReduce(double4 *d_in, int N);

enum InteractionType_t {LJ, LJ_SMOOTH, INTERACTION_TYPE_UNDEFINED};

__constant__ double dev_strength;

//__global__ void populate_cell_nebz(uint2 *c, uint4 *cn1, uint4 *cn2);
//__global__ void get_cid(const double4 __restrict__ *r, long *cid, int *clen) ;
//__device__ void get_pair_lj_(const double4 __restrict__ *r_i,const double4 __restrict__ r_j,double4 *f_i,double4 *pe,double4 *virial_ij,double3 *born_ij);
/*
//template <int BLOCK_THREADS>
//__global__ void calculate_force_with_cell_list(const double4 __restrict__ *r_,
									double4 *f_,
									const long  *pid_,
									const long  *cid_,
									const int  *clen,
									const int  *cbegin,
									const uint4 __restrict__ *cl_neb1,
									const uint4 __restrict__ *cl_neb2,
									double* dev_pe_reduced,        // pe, [reduced over blocks]
                             		double4* dev_virial_tensor_pp,     // 4 component virial tensor [per particle]
                             		double4* dev_virial_tensor_reduced, // 4 compoent virial tensor [reduced over blocks]
                             		double* dev_virial_pressure_reduced     // sigma = sigma_xx + sigma_yy [reduced over blocks]
									) ;
*/

class NonBondedForces {
private:
	InteractionType_t interaction_type;
	double sigma;
	double epsilon;
	double rc,rc2;
	
	//Cell_List arrays
	uint2 *dev_cell_list; // x->start_id, y->len;
	uint4 *dev_cell_nebz1; // E NE N NW
	uint4 *dev_cell_nebz2; // W SW S SE
	int *dev_clen;
	int *dev_cbegin;
	
	uint2 *host_cell_list; // x->start_id, y->len;
	uint4 *host_cell_nebz1; // E NE N NW
	uint4 *host_cell_nebz2; // W SW S SE
	int *host_clen;
	int *host_cbegin;
	
	//rc & cell
	uint3 ncell; //x,y,z=x*y
	double3 l;
	double3 lcell;
	
	// Strength default to 1.0
	double strength;
	
	void dev_allocate_celllist();
	void host_allocate_celllist();
	void copy_celllist_d2h();
	void host_set_cell_params();
	void sort_by_cid();
		
public:
	int interpret(std::vector<std::string> tokenized);
	void set_box_dim(const double &lx, const double &ly);
	void compute();

	//compute forces
	void calculate_non_bonded_force();

	//debug
	//void print_celllist();
	
	//void reduce_thermo_data();






};
#endif
