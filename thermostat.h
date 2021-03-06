#ifndef __THERMO_H__
#define __THERMO_H__ 

#include <sstream>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <iterator>
#include <cmath>

#include <boost/lexical_cast.hpp>

#include "vector_types.h"
#include "global.h"
#include "cuda_constants.cuh"
#include "kernels/sum4.h"

#include <cub/cub.cuh>

#include "ThermostatBDP.h"

extern double GlobalReduce(double *d_in, int N);
extern double4 GlobalReduce(double4 *d_in, int N);

enum Thermostat_t {BERENSDEN, BDP, LANGEVIN, NO_THERMOSTAT};

//double *dev_v2_reduced, 
//double4 *dev_v_tensor_reduced, 

class Thermostat {
private:
	Thermostat_t thermostat_type;
	double lambda;
	double target_temperature, tau_t;
	
	ThermostatBDP *tstat;
	
	// ke, ke_temsor, temperature
	double4 v_tensor;
	double ke, temperature;

public:
	void apply(); // rescale velocity + measure ke
	void init_velocities();
	
	void interpret(vector<std::string> &tokenized);
	
	// getter 
	double get_ke();
	double get_temperature();
	double4 get_v_tensor();

};






#endif 
