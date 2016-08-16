#include "thermostat.h"

// ************************ KERNEL BEGIN ****************************//
template <int BLOCK_THREADS> __global__ 
void kernel_rescale_velocities(double4* __restrict__ v_, 
							   double scale_factor,
							   double *dev_v2_reduced, 
							   double4 __restrict__ *dev_v_tensor_reduced, 
							   int mode, // 1: measure ke tensor
							   long N) {

	int gtid = blockDim.x*blockIdx.x + threadIdx.x;
	if(gtid >= N) return;

	//read to registers
	double4 v = v_[gtid];
	v.x*=scale_factor;
	v.y*=scale_factor;

	//write back to global mem
	v_[gtid] = v;

	typedef cub::BlockReduce<double4, BLOCK_THREADS, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduceDouble4;
	__shared__ typename BlockReduceDouble4::TempStorage temp_storage_double4;
	double4 reduced;

	//thermo - temperature
	double4 v2 = make_double4(v.x*v.x + v.y*v.y, 0.0, 0.0, 0.0);
	reduced = BlockReduceDouble4(temp_storage_double4).Reduce(v2, Sum4());
	if(threadIdx.x == 0) dev_v2_reduced[blockIdx.x] = reduced.x;

	//if(mode == 1) {
		//thermo - ke tensor
		double4 my_v_tensor = make_double4(v.x*v.x, v.x*v.y, v.y*v.x, v.y*v.y);
		reduced = BlockReduceDouble4(temp_storage_double4).Reduce(my_v_tensor, Sum4());
		if(threadIdx.x == 0) dev_v_tensor_reduced[blockIdx.x] = reduced; //todo
	//}

}

// ************************ KERNEL END ****************************//

void Thermostat::interpret(vector<std::string> &tokenized) {
	std::cout << std::endl << "|---------------- Thermostat -----------------|" << std::endl;	

	if(tokenized[1] == "BDP") {
		thermostat_type = BDP;
		tstat = new ThermostatBDP;
		target_temperature = boost::lexical_cast<double>(tokenized[2]);
		tau_t = boost::lexical_cast<double>(tokenized[3]);
		
		tstat->set(tau_t/dt);
		
		std::cout << "target_temp = " << target_temperature << " " << "tau_t = " << tau_t << std::endl;
		
	} else if (tokenized[1] == "Berensden") {
		thermostat_type = BERENSDEN;
		target_temperature = boost::lexical_cast<double>(tokenized[2]);
		tau_t = boost::lexical_cast<double>(tokenized[3]);
		std::cout << "Using Berensden thermostat" << std::endl;
		std::cout << "target_temp = " << target_temperature << " " << "tau_t = " << tau_t << std::endl;
	
	} else if(tokenized[1] == "no") {
		thermostat_type = NO_THERMOSTAT;
		//TODO
		target_temperature = boost::lexical_cast<double>(tokenized[2]);
		lambda = 1.0;
		std::cout << "Thermostat: No thermostat - NVE ensemble" << std::endl;
	
	} //else if(tokenized[1] == "BDP") //handle other cases 
	
	else {
		std::cout << "thermostat not recognized" << std::endl;
	}

}

void Thermostat::apply() { 

	if(thermostat_type == BDP) {
		kernel_rescale_velocities<128><<<nblocks, nthreads>>>(dev_v, 1.0, dev_v2_reduced, dev_v_tensor_reduced, 0, N);
		double old_ke = 0.5*GlobalReduce(dev_v2_reduced, nblocks);
		double new_ke = tstat->resamplekin(old_ke, N*target_temperature, 2*N);
		lambda = std::sqrt(new_ke / old_ke);
	} 
	
	if(thermostat_type == BERENSDEN) {
		kernel_rescale_velocities<128><<<nblocks, nthreads>>>(dev_v, 1.0, dev_v2_reduced, dev_v_tensor_reduced, 0, N);
		double now_temperature = GlobalReduce(dev_v2_reduced, nblocks)/double(2.0*N - 3.0);
		 lambda = sqrt(1.0 + (dt/tau_t)*(target_temperature/now_temperature - 1.0));
	}
	
	else {
		lambda = 1.0;
	}
	
	kernel_rescale_velocities<128><<<nblocks, nthreads>>>(dev_v, lambda, dev_v2_reduced, dev_v_tensor_reduced, 1, N);
	double sum_v2 = GlobalReduce(dev_v2_reduced, nblocks);
	temperature = sum_v2/double(2*N - 3.0);
	ke = 0.5*sum_v2;	
}


double Thermostat::get_ke() {return ke;}
double Thermostat::get_temperature() {return temperature;}
double4 Thermostat::get_v_tensor() {return v_tensor;}

void Thermostat::init_velocities() {
	std::cout << "Initializing velocities for temperature = " << target_temperature << std::endl;
	
	//Take away CM drift
	double cmvx = 0.0;
	double cmvy = 0.0;

	for (unsigned int i = 0; i < N; i++) {
		cmvx += host_v[i].x;
		cmvy += host_v[i].y;;
	}
	cmvx /= double(N);
	cmvy /= double(N);

	double sumv2 = 0.0;
	for (unsigned int i = 0; i < N; i++) {
		host_v[i].x -= cmvx;
		host_v[i].y -= cmvy;
		sumv2 += host_v[i].x * host_v[i].x + host_v[i].y * host_v[i].y;
	}

	//rescale to target_T
	double temp = sumv2 / double(2*N - 3.0); // REVIEW THIS 4
	double fac = sqrt(target_temperature / temp);

	for (unsigned int i = 0; i < N; i++) {
		host_v[i].x *= fac;
		host_v[i].y *= fac;
	}
}



