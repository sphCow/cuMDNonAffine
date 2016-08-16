#ifndef __THERMO_H__
#define __THERMO_H__ 

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
#include "kernels/sum4.h"

#include <cub/cub.cuh>

extern double GlobalReduce(double *d_in, int N);
extern double4 GlobalReduce(double4 *d_in, int N);

class Thermo {
private:

	






}






#endif 
