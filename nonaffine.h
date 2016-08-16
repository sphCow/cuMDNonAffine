#ifndef NON_AFFINE
#define NON_AFFINE

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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

#include "kernels/sum4.h"

__global__ void kernel_calc_X(double4 *r, long *pid, long *nebz_pid,
                              double4 *ref_r ,long *pid2gtid, double4 *X ,
                              long N, int num_nebz);

__device__ inline double d_pbcx(double xx);
__device__ inline double d_pbcy(double yy);

extern "C" void map_pid2gtid(long *dev_pid,long *dev_map_pid2gtid,const long &N ,const int &nblocks,const int &nthreads);

class NonAffine {
private:
  //TODO
  long N;

  double4 *X;
  double4 *Y;
  double4 *epsilon;
  double4 *ref_r;
  long *ref_pid;
  double *chi2;

  //gpu arrays
  double4 *dev_X;
  double4 *dev_Y;
  double4 *dev_epsilon;
  double4 *dev_ref_r;
  //long *ref_pid;
  
  double *dev_chi2;
  double *dev_chi2_sqr;

  // arrays copied from gpu to cpu
  double* chi2_gpu;

	// DEBUG HCHI FORCES
	double4 *dev_f_hchi;
	double4 *f_hchi;

  //mean
  double4 mean_epsilon;
  double mean_chi2;
  double mean_chi2_sqr;

  double lx, ly;


  std::ofstream file_ref;
  std::ofstream file_out;
  std::ofstream file_per_part;


  std::string ref_file_name;
  void allocate();
  int read_reference_from_file();

  //**************** NEBZ *************** //
  //populate nebz (triangular only)
  static const int num_nebz = 6; //HARD CODED

  long *nebz_pid;
  long *dev_nebz_pid;

  void populate_nebz();

  void calc_X(double4 *r);

  void calc_Y(); //only uses reference pos
  void calc_epsilon();

  //******** FORCE CALCULATION FOR H_chi *******//	
  bool calc_force_hchi_required;
  void calc_force_hchi();
  double h;

  //*********** PBC ********************* //
  // TODO make pbc globally available
  double pbcx(double );
  double pbcy(double );




public:
  NonAffine(void);
  ~NonAffine();

  //compute with CPU/GPU
  UseDevice use_device;

  //setter
  void set_box_dim(double lx, double ly);
  void scale_ref_lattice(double, double); //
  void set_N_read_ref(long N_);

  //getter
  double get_mean_chi2() {return mean_chi2;}
  double get_mean_chi2_sqr() {return mean_chi2_sqr;}
  double4* get_ref_lattice();

  //initialization
  void interpret(std::vector<std::string> tokenized);

  //calculation
  void calc_chi2(double4* r);
  void get_pp_chi2(); //only required for gpu

  //ouput
  // void open_file(std::string prefix);
  // void write(long t, int num_deformed, double4* stress, std::string s);






};
#endif
