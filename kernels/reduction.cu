#ifndef __REDUCTION_CU__
#define __REDUCTION_CU__

#include "sum4.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

extern double GlobalReduce(double *d_in, int N) {
  double h_out;

  thrust::device_ptr<double> ptr1(d_in);
  thrust::device_ptr<double> ptr2(d_in+N);
  h_out = thrust::reduce(ptr1, ptr2);
  
  return h_out;
}

extern double4 GlobalReduce(double4 *d_in, int N) {
  double4 h_out;
  
  thrust::device_ptr<double4> ptr1(d_in);
  thrust::device_ptr<double4> ptr2(d_in+N);
  double4 init = make_double4(0.0,0.0,0.0,0.0);
  h_out = thrust::reduce(ptr1, ptr2, init,Sum4());
  
  return h_out;
}



#endif
