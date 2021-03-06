#ifndef __BONDED_FORCES__H
#define __BONDED_FORCES__H


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

#include "kernels/sum4.h"

extern double GlobalReduce(double *d_in, int N);
extern double4 GlobalReduce(double4 *d_in, int N);
extern "C" void map_pid2gtid(long *dev_pid,long *dev_map_pid2gtid,const long &N ,const int &nblocks,const int &nthreads);

enum BondType_t {HARMONIC, BONDTYPE_UNDEFINED};
enum NebType_t {TRIANGULAR, NEBTYPE_UNDEFINED};

class BondedForces {
private:

  BondType_t bond_type;
  NebType_t neb_type;

  double k;
  double r0; //eq bond length

  long Nx,Ny;
  double lx,ly;

  // Nebz list
  int num_nebz;
  long *nebz_pid;
  long *dev_nebz_pid;

public:
  int interpret(std::vector<std::string> tokenized);
  void init();
  void set_box_dim(const double &lx, const double &ly);

  //compute forces
  void compute();


};

#endif
