////////////////////////////////////////////////////////////////////////////////////////////////
// Let it be a well-optimized GPU implementation of Molecular Dynamicss		  //
// in my one way. It might not be the fastest, might not be implemented           //
// in the most standard way, but It'd be "mine", my very own implementation   //
// of massively parallel MD in GPU. So, with love and passion towards physics //
// and computer programming, lets get started...                                                          //
//																			  //
// Author -> Parswa Nath [TIFR-TCIS]										  //
// Use & redistribute as you wish, I don't care                                                           //
////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <assert.h>
#include <vector>
#include <map>
#include <boost/lexical_cast.hpp>
#include <boost/assign.hpp>


#include "global.h"
#include "cuda_constants.cuh"
#include "dev_thermo_array.cuh"

#include "nonaffine.h"
#include "BondedForces.h"
#include "NonBondedForces.h"
#include "thermostat.h"

//#include "class_ptrs.h"
//kernels
//#include "kernels/map_pid2gtid.cu"

using namespace boost::assign;

#define dim 2;

double rho;
//double dt;

//rc & cell
double rc;
double3 l;

//strain
double strain_rate;
int num_of_strain_applied = 0;
bool is_strain_data_loaded = false;
unsigned long strain_apply_interval = 0;
double e_bar = 0.0;
double epsilon_now = 0.0;
double epsilon_initial = 0.0;

// *********** init classes *************** //
	NonAffine *nonAffine;
	bool is_NonAffine_inilialized = false;

	BondedForces *bondedForces;
	bool is_BondedForces_initialized = false;

	NonBondedForces *nonBondedForces;
	bool is_NonBondedForces_initialized = false;
	
	Thermostat *thermostat = new Thermostat;
// **************************************** //

// Kernel launch config
// long nblocks, nthreads;
int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

//****************** EXTERN PROTOTYPES ***************************//
extern double GlobalReduce(double *d_in, int N);
extern double4 GlobalReduce(double4 *d_in, int N);
extern void set_accumulators_zero();
//****************************************************************//

// GPU info
int get_gpu_info()
{
	int nDevices;

	printf("# ---------- GPU INFO ----------- #\n");

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("GPU- %d -> ", i);
		printf("  Device name: %s\n", prop.name);
		//printf("  Memory Clock Rate (KHz): %d\n",
		//       prop.memoryClockRate);
		//printf("  Memory Bus Width (bits): %d\n",
		//       prop.memoryBusWidth);
		//printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
		//       2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
	}
	
	return nDevices;
}

// Generate lammps data
std::string generate_lammps_data(double4 *r, double4 *v, long N, double lx, double ly)
{
	std::stringstream ss;

	ss << "LAMMPS Description" << std::endl;
	ss << std::endl;

	ss << N << " atoms" << std::endl;
	ss << "0 bonds\n0 angles\n0 dihedrals\n0 impropers" << std::endl;
	ss << std::endl;

	ss << "1 atom types" << std::endl << std::endl;

	ss << std::setprecision(16) << -lx / 2.0 << " " << std::setprecision(16) << lx / 2.0 << " xlo xhi" << std::endl;
	ss << std::setprecision(16) << -ly / 2.0 << " " << std::setprecision(16) << ly / 2.0 << " ylo yhi" << std::endl;
	ss << "-0.5 0.5 zlo zhi" << std::endl << std::endl;

	ss << "Masses" << std::endl << std::endl;
	ss << "1 1.0" << std::endl << std::endl;

	ss << "Pair Coeffs" << std::endl << std::endl;
	ss << "1 1.0 1.0" << std::endl << std::endl;

	ss << "Atoms" << std::endl << std::endl;

	for (int i = 0; i < N; i++)
		ss << i + 1 << " " << "1 " << std::setprecision(16) << r[i].x << " " << std::setprecision(16) << r[i].y << " 0.000" << std::endl;

	ss << std::endl;
	ss << "Velocities" << std::endl << std::endl;

	for (int i = 0; i < N; i++)
		ss << i + 1 << " " << std::setprecision(16) << v[i].x << " " << std::setprecision(16) << v[i].y << " 0.000" << std::endl;

	return ss.str();
}

/////////////////////////////////////////////////////////////////////////////
//				Allocation/Data transfer/Print                             //
// http://stackoverflow.com/questions/31598021/cuda-cudamemcpy-struct-of-arrays/31599025#comment51148602_31598021
/////////////////////////////////////////////////////////////////////////////
/***** Lattice *****/
void host_allocate_lattice(long N)
{
	host_pid = new long[N]();
	host_cid = new long[N]();
	host_r = new double4[N]();
	host_v = new double4[N]();
	host_f = new double4[N]();
	host_virial_tensor_pp = new double4[N]();
	host_pid2gtid = new long[N]();
	printf("[HOST] allocated memory for lattice\n");
}

void dev_allocate_lattice(long N)
{
	cudaMalloc((void **)&dev_pid, N * sizeof(*dev_pid));
	cudaMalloc((void **)&dev_cid, N * sizeof(long));
	cudaMalloc((void **)&dev_r, N * sizeof(*dev_r));
	cudaMalloc((void **)&dev_v, N * sizeof(*dev_v));
	cudaMalloc((void **)&dev_f, N * sizeof(*dev_f));
	cudaMalloc((void **)&dev_virial_tensor_pp, N * sizeof(*dev_virial_tensor_pp));
	cudaMalloc((void **)&dev_pid2gtid, N * sizeof(*dev_pid2gtid));
	
	//printf("[DEVICE] allocated memory for lattice\n");
}

void copy_lattice_h2d()
{
	cudaMemcpy(dev_pid, host_pid, N * sizeof(*dev_pid), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cid, host_cid, N * sizeof(*dev_cid), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r, host_r, N * sizeof(*dev_r), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v, host_v, N * sizeof(*dev_v), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_f, host_f, N * sizeof(*dev_f), cudaMemcpyHostToDevice);
}

void copy_lattice_d2h()
{
	cudaMemcpy(host_pid, dev_pid, N * sizeof(*host_pid), cudaMemcpyDeviceToHost);
	//cudaMemcpy(host_cid, dev_cid, N*sizeof(*host_cid), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_r, dev_r, N * sizeof(*host_r), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_v, dev_v, N * sizeof(*host_v), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_f, dev_f, N * sizeof(*host_f), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_virial_tensor_pp, dev_virial_tensor_pp, N * sizeof(*host_virial_tensor_pp), cudaMemcpyDeviceToHost);

	//debug
	//cudaMemcpy(host_pid2gtid, dev_pid2gtid, N * sizeof(*host_pid2gtid), cudaMemcpyDeviceToHost);
}


__host__ void sort_host_arrays_by_pid()
{
	//sort by pid
	// thrust::sort_by_key(host_pid, host_pid+N,
	//      thrust::make_zip_iterator(thrust::make_tuple(
	//              host_cid, host_r, host_v, host_f, host_virial_tensor_pp)));

	thrust::sort_by_key(host_pid, host_pid + N,
			    thrust::make_zip_iterator(thrust::make_tuple(
							      host_r, host_v, host_virial_tensor_pp)));
}

__host__ void sort_dev_arrays_by_pid_and_copy() {
	thrust::device_ptr<long> dev_ptr_pid = thrust::device_pointer_cast(dev_pid);
	//thrust::device_ptr<long> dev_ptr_cid = thrust::device_pointer_cast(dev_cid);
	thrust::device_ptr<double4> dev_ptr_r = thrust::device_pointer_cast(dev_r);
	thrust::device_ptr<double4> dev_ptr_v = thrust::device_pointer_cast(dev_v);
	//thrust::device_ptr<double4> dev_ptr_f = thrust::device_pointer_cast(dev_f);
	thrust::device_ptr<double4> dev_ptr_virial_tensor_pp = thrust::device_pointer_cast(dev_virial_tensor_pp);

	thrust::sort_by_key(dev_ptr_pid, dev_ptr_pid + N,
		thrust::make_zip_iterator(
			thrust::make_tuple(dev_ptr_r, dev_ptr_v, dev_ptr_virial_tensor_pp)));
			
	copy_lattice_d2h();
}

/*
std::string make_pp_header(long t, long tsteps_max)
{
	std::stringstream ss;

	//make header TODO
	std::map<string, double> info = map_list_of("current_tstep", t)
						("lx", l.x)("ly", l.y)("n", N)
						("num_of_strain_applied", num_of_strain_applied)
						("strain_rate", strain_rate)("tstep", tsteps_max);

	std::map<string, double>::iterator it;

	ss << "# ";
	for (it = info.begin(); it != info.end(); it++) ss << it->first << " ";
	ss << std::endl;

	ss << "# ";
	for (it = info.begin(); it != info.end(); it++) ss << std::setprecision(16) << double(it->second) << " ";
	ss << std::endl << std::endl;

	return ss.str();
}
*/

void dev_allocate_thermo()
{
	cudaMalloc((void **)&dev_pe_reduced, nblocks * sizeof(double));
	cudaMalloc((void **)&dev_virial_tensor_reduced, nblocks * sizeof(double4));
	cudaMalloc((void **)&dev_virial_pressure_reduced, nblocks * sizeof(double));
	

	cudaMalloc((void **)&dev_v2_reduced, nblocks * sizeof(*dev_v2_reduced));						   
	cudaMalloc((void **)&dev_v_tensor_reduced, nblocks * sizeof(*dev_v_tensor_reduced));
}



/////////////////////////////////////////////////////////////////////////////
// [CPU] Generate Lattice									   //
// sets l.x,ly
// TODO : sample velocity from Gaussian dist.                  //
/////////////////////////////////////////////////////////////////////////////
__host__ void generate_lattice(long Nx, long Ny, int type)
{
	double ax, ay;

	if (type == 0) {
		printf("# generating Triangular lattice %ld %ld %lf \n", Nx, Ny, rho);
		ax = sqrt(2.0 / (rho * sqrt(3.0)));
		ay = ax * sqrt(3.0) / 2.0;
	} else if (type == 1) {
		printf("# generating square lattice\n");
		ax = sqrt(0.0 / rho);
		ay = ax;
	} else {
		fprintf(stderr, "unknown lattice!\n");
	}

	l.x = Nx * ax;
	l.y = Ny * ay;

	double lxh = l.x / 2.0;
	double lyh = l.y / 2.0;

	srand48(100);

	for (long i = 0; i < Nx; i++) {
		for (long j = 0; j < Ny; j++) {
			double xx = -lxh + i * ax;
			double yy = -lyh + j * ay;

			if (type == 0 && j % 2 != 0)
				xx += ax / 2.0;

			long id = i + j * Nx;

			host_pid[id] = id;
			host_r[id].x = xx;              //x
			host_r[id].y = yy;              //y
			host_r[id].z = 0.0;             //phi //TODO

			host_v[id].x = drand48() - 0.5; //x //TODO
			host_v[id].y = drand48() - 0.5; //y //TODO
			host_v[id].z = 0.0;             //phi //TODO

			host_f[id].x = 0.0;             //x
			host_f[id].y = 0.0;             //y
			host_f[id].z = 0.0;             //phi
		}
	}
}




////////////////////////////////////////////////////////////////////////////////
// Reduce over threads kernel                                                 //
//
////////////////////////////////////////////////////////////////////////////////
__device__ void reduce_over_thread(volatile double *sdata, double *g_odata)
{
	unsigned int tid = threadIdx.x;

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	//__syncthreads();
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
/////////////////////////////////////////////////////////////////////////////////

__global__ void preforce_velocity_verlet(double4 *r_, double4 *v_, double4 *f_) {
	int gtid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gtid >= dev_N) return;

	//read to registers
	double4 r = r_[gtid];
	double4 v = v_[gtid];
	double4 f = f_[gtid];
	double dt = dev_dt;

	//full update r
	r.x += dt * v.x + 0.5 * dt * dt * f.x;
	r.y += dt * v.y + 0.5 * dt * dt * f.y;

	//pbc
	r.x -= dev_l.x * rint(r.x / dev_l.x);
	r.y -= dev_l.y * rint(r.y / dev_l.y);

	//DEBUG
	//uint my_cx = uint((p->r[gtid].x + dev_l.x/2.0) / dev_lcell.x);
	//uint my_cy = uint((p->r[gtid].y + dev_l.y/2.0) / dev_lcell.y);
	//uint my_cid = (my_cx) + (my_cy) * (dev_ncell.x);
	//p->cid[gtid] = my_cid;

	//half update v
	v.x += 0.5 * dt * f.x;
	v.y += 0.5 * dt * f.y;

	//write r & half updated v to global mem
	r_[gtid] = r;
	v_[gtid] = v;
}


__global__ void postforce_velocity_verlet(double4 *v_, double4 *f_)
{
	//extern __shared__ double shared_thermo[];

	int gtid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gtid >= dev_N) return;

	//read to registers
	double4 v = v_[gtid];
	double4 f = f_[gtid];
	double dt = dev_dt;

	//full update velocity
	v.x += 0.5 * dt * f.x;
	v.y += 0.5 * dt * f.y;

	//write to global mem
	v_[gtid] = v;
}

/* RESCALE PARTICLE POSITIONS */
__global__ void rescale_particle_pos(double4 *r_, double scale_factor)
{
	int gtid = blockDim.x * blockIdx.x + threadIdx.x;

	if (gtid >= dev_N) return;

	//read to registers
	double4 r = r_[gtid];
	r.x = (1.0 + scale_factor) * r.x;
	r.y = (1.0 - scale_factor) * r.y;

	//write back to global mem
	r_[gtid] = r;
}

void write_restart_data(long N, std::string file_name) {

	copy_lattice_d2h();
	sort_host_arrays_by_pid();
	//sort_dev_arrays_by_pid_and_copy();
	
	// Map of {key, vals} which have to be written to restart file 
	map<string, double> info = map_list_of("lx", l.x)("ly", l.y)("n", N)("dt", dt)
					    ("nblocks", nblocks)("nthreads", nthreads)
					    ("tsteps_now", tstep)
					    ("tsteps_max", tsteps_max)
					    ("strain_apply_interval", strain_apply_interval)
							("strain_rate", strain_rate)
							("e_bar", e_bar)
							("num_of_strain_applied", num_of_strain_applied)
							("epsilon_now", epsilon_now);
							

	ofstream f;
	std::string name = file_name + ".restart.out";
	f.open(name.c_str());

	// Write info 
	f << "Begin_Info" << std::endl;
	std::map<string, double>::iterator it;
	for (it = info.begin(); it != info.end(); it++) f << it->first << " ";
	f << std::endl;
	for (it = info.begin(); it != info.end(); it++) f << std::setprecision(16) << it->second << " ";
	f << std::endl;

	// vector<string> header = {"id", "Pe", "x", "y","z",
	//                          "vx","vy","vz",
	//                          "fx", "fy", "fz"};

	// write particle data
	f << std::endl << "Begin_Particle_Data" << std::endl;

	for (long i = 0; i < N; i++) {
		f	<< host_pid[i] << " "
			<< host_r[i].x << " " << host_r[i].y << " " << host_r[i].z << " "
			<< host_v[i].x << " " << host_v[i].y << " " << host_v[i].z << std::endl;
	}

	std::cout << "restart data written to " << file_name << ".restart.out" << std::endl;

	f.close();
	
	//[NEW] write nonaffine ref lattice to file 
	if(is_NonAffine_inilialized) {
		double4* ref;
		ref = new double4[N];
		ref = nonAffine->get_ref_lattice();
		
		std::ofstream ss;
		std::string name = file_name+".ref.pplattice";
		ss.open(name.c_str());
			ss << "# NEW_FRAME current_tstep lx ly N num_strain_applied strain_rate strain_apply_interval" << std::endl;
			ss << tstep << " " << l.x << " " <<  l.y << " " << N << " " << num_of_strain_applied << " " << strain_rate << " " << strain_apply_interval << std::endl;
			
			ss << "#        pid          r[0]          r[1]" << std::endl;
			//fileIO->write_per_atom_qty(ss.str());
			for(int i=0; i<N; i++) {
				ss << i << "\t" << ref[i].x << "\t" << ref[i].y << std::endl;
			}
			
			std::cout << "data for nonaffine ref lattice is written to " << file_name << ".ref.pplattice" << std::endl;
			ss.close();
			delete[] ref;
	}
	
	
}

int load_lattice(std::map<string, double> &info, string filename)
{
	ifstream f;

	f.open(filename.c_str());

	if (!f.good()) return -1;

	string line;
	long lcount = 0;
	long header_begin = -1;
	long lattice_begin = -1;
	long particle_count = 0;

	std::vector<string> header_key;
	std::vector<string> header_val;

	while (getline(f, line)) {
		if (line == "Begin_Info") {
			header_begin = lcount;
		} else if (line == "Begin_Particle_Data") {
			lattice_begin = lcount + 1;

			for (int i = 0; i < header_key.size(); i++)
				// info.emplace(header_key[i], boost::lexical_cast<double>(header_val[i]));
				info[header_key[i]] = boost::lexical_cast<double>(header_val[i]);

			// ***** ALLOCATION HERE ******//
			N = boost::lexical_cast<long>(info["n"]);
			std:: cout << "from load " << N << std::endl;
			host_allocate_lattice(N);
			dev_allocate_lattice(N);
		}

		if (lcount == (header_begin + 1)) {
			stringstream iss(line);
			copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(header_key));
		}

		if (lcount == (header_begin + 2)) {
			stringstream iss(line);
			copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(header_val));
		}

		// LATTICE
		if (lattice_begin != -1 && lcount >= lattice_begin) {
			stringstream iss(line);

			iss >> host_pid[particle_count]
			>> host_r[particle_count].x
			>> host_r[particle_count].y
			>> host_r[particle_count].z
			>> host_v[particle_count].x
			>> host_v[particle_count].y
			>> host_v[particle_count].z;

			particle_count++;
		}

		lcount++;
	}

	// for(int i=0; i<header_key.size(); i++) {
	//     info.emplace(header_key[i], boost::lexical_cast<double>(header_val[i]));
	// }

	//check # of particles
	if (N != long(info["n"])) {
		cerr << "Number of particles doesn't match" << endl;
		return -1;
	}

	return 0;
}

int main(int argc, char **argv)
{
	//*fileIO = new FileIO();

	int gpuid = 0;

	if (argc == 2)
		gpuid = boost::lexical_cast<int>(argv[1]);

	cudaSetDevice(gpuid);
	std::cout << "CUDA Device set to GPU-" << gpuid << std::endl;


	//FileIO* fileIO;

	enum lattice_init_t { LATTICE_NOT_DEFINED, NEW, LOAD };
	enum integrator_init_t { INTEGRATOR_NOT_DEFINED, VELOCITY_VERLET, LANGEVIN, NONE };

	lattice_init_t lattice_init = LATTICE_NOT_DEFINED;
	integrator_init_t integrator_init = INTEGRATOR_NOT_DEFINED;

	long Nx, Ny;

	/***** Interpret inputs ******/
	//unsigned long strain_apply_interval = 0;    // = 0;
	unsigned long thermo_save_interval;     //= 100;
	unsigned long config_save_interval;     //= 0;
	//double e_bar = 0.0;
	std::string prefix;                     //= "just_another_md";
	std::string lattice_load_filename;

	// STOP AT STRAIN
	double max_epsilon = -1; // 2*nd*e_bar

	std::string line;
	while (std::getline(cin, line)) {
		std::vector<std::string> tokenized;
		if (line[0] == '#' || line.empty()) continue; //skip empty lines && lines starting with #
		fileIO->tokenize(line, tokenized);

		// prefix
		if (tokenized[0] == "prefix") {
			prefix = tokenized[1];
			std::cout << "prefix set to -> " << prefix << std::endl;
			//fileIO = new FileIO;
			fileIO->set_file_prefix(prefix);
		} 
		
		// non-bonded forces
		else if (tokenized[0] == "non_bonded_force") {
			nonBondedForces = new NonBondedForces;
			is_NonBondedForces_initialized = true;
			nonBondedForces->interpret(tokenized);
		}
		
		// lattice
		else if (tokenized[0] == "lattice" && lattice_init == LATTICE_NOT_DEFINED) {
			if (tokenized[1] == "new") {
				Nx = boost::lexical_cast<long>(tokenized[3]);
				Ny = boost::lexical_cast<long>(tokenized[4]);
				N = Nx * Ny;
				rho = boost::lexical_cast<double>(tokenized[5]);
				lattice_init = NEW;
			} else if (tokenized[1] == "load") {
				std::cout << "lattice to be loaded from " << tokenized[2] << std::endl;
				lattice_init = LOAD;
				lattice_load_filename = tokenized[2];
			} else {
				std::cerr << "Unknown lattice args" << std::endl;
			}
		}
		//integrator
		else if (tokenized[0] == "integrator" && integrator_init == INTEGRATOR_NOT_DEFINED) {
			if (tokenized[1] == "velocity_verlet") {
				std::cout << "integrator -> " << "velocity_verlet" << std::endl;
				integrator_init = VELOCITY_VERLET;
				dt = boost::lexical_cast<double>(tokenized[2]);
				tsteps_max = boost::lexical_cast<long>(tokenized[3]);
			} else if (tokenized[1] == "langevin") {
				integrator_init = LANGEVIN;
				std::cout << "integrator -> " << "langevin" << std::endl;
			} else if (tokenized[1] == "none") {
				integrator_init = NONE;
				std::cout << "integrator -> " << "NO INTERGRATOR" << std::endl;
			} else {
				std::cerr << "unknown integrator" << std::endl;
			}
		}
		//strain
		else if (tokenized[0] == "strain") {
			std::cout << "|---------------- Strain ----------------|" << std::endl;
			strain_apply_interval = boost::lexical_cast<long>(tokenized[1]);
			strain_rate = boost::lexical_cast<double>(tokenized[2]); // this is e
			e_bar = strain_rate/strain_apply_interval;
			std::cout << "strain_apply_interval =  " << strain_apply_interval << std::endl;
			std::cout << "shear_step            =  " << strain_rate << std::endl;
			std::cout << "e_bar                 =  " << e_bar << std::endl;
			std::cout << "e_bar (in LJ units)   =  " << e_bar/dt << std::endl;
			
			//if(is_strain_data_loaded == false) num_of_strain_applied = 0;
			//else {
			//	
			//}
			
			std::cout << "|---------------- Strain ----------------|" << std::endl;
		}
		
		// thermostat
		else if (tokenized[0] == "thermostat") {
			thermostat->interpret(tokenized);
		}
		
		//NonAffine
		else if (tokenized[0] == "non_affine") {
			nonAffine = new NonAffine;
			is_NonAffine_inilialized = true;
			nonAffine->interpret(tokenized);
		}
		
		//BondedForces
		else if (tokenized[0] == "bonded_force") {
			bondedForces = new BondedForces;
			is_BondedForces_initialized = true;
			bondedForces->interpret(tokenized);
		}
		
		//save
		else if (tokenized[0] == "save") {
			thermo_save_interval = boost::lexical_cast<long>(tokenized[1]);
			config_save_interval = boost::lexical_cast<long>(tokenized[2]);
		} 
		
		
		//stop at 
		else if(tokenized[0] == "stop_at") {
			if(tokenized[1] == "epsilon" ) {
				max_epsilon = boost::lexical_cast<double>(tokenized[2]);
				std::cout << "stop_at max epsilon " << max_epsilon << std::endl;
				
				//-------- calc max deformations required && tmax -----------//
				int Nd_max = max_epsilon/(2.0*strain_rate);
				int t_max_to_reach_epsilon_max = max_epsilon/(2.0*e_bar);
				std::cout << "|-------- Runtime estimate to reach max strain --------|" << std::endl;
				std::cout << "|-- max_epsilon = " << max_epsilon << std::endl;
				std::cout << "|-- Nd_max      = " << Nd_max << std::endl;
				std::cout << "|-- t_max       = " << t_max_to_reach_epsilon_max << std::endl;
				std::cout << "|------------------------------------------------------|" << std::endl;
				
			} //else if()
			
			else {
				std::cout << "unknown stop_at keyword" << std::endl;
			}
		}
		
		//restart data
		else if(tokenized[0] == "write_restart_data") { 
			if(tokenized[1] == "every") {
				write_restart_data_interval = boost::lexical_cast<int>(tokenized[2]);
				restart_data_count = 0;
				std::cout << "Restart Data:: interval = " << write_restart_data_interval << std::endl;
			} 
			
			else {
				std::cout << "Unknown write_restart_data " << std::endl;
			}
		}
		
		else {
			std::cout << "ignoring unrecognized/duplicate keyword(s) " << tokenized[0] << endl;
		}
	
	}

	/********** Files ***********/
	std::ofstream file_thermo;
	//std::ofstream file_lattice;
	std::ofstream file_log;

	std::string file_thermo_name = prefix + ".thermo";
	//std::string file_lattice_name = prefix+".lattice";
	std::string file_log_name = prefix + ".log";

	file_thermo.open(file_thermo_name.c_str());
	file_log.open(file_log_name.c_str());
	//if(config_save_interval>0)
	//file_lattice.open(file_lattice_name.c_str());

	// ****************** generate or load lattice/restart file ***************** //
	std::map<std::string, double> info;

	switch (lattice_init) {
	case NEW:
		std::cout << "|----------------- Lattice Generation -----------------|" << std::endl;
		host_allocate_lattice(N);
		dev_allocate_lattice(N);
		generate_lattice(Nx, Ny, 0);
		thermostat->init_velocities();

		//nblocks = Nx;
		//nthreads = Ny;

		nblocks = iDivUp(N, 128); //TODO
		nthreads = 128;
		
		//nblocks = iDivUp(N, 64); //TODO
		//nthreads = 64;

		break;

	case LOAD:
		int load_result = load_lattice(info, lattice_load_filename);

		if (load_result != 0) {
			std::cerr << "can't load lattice" << std::endl;
			return -1;
		}

		//if loaded successfully DEBUG
		std::cout << "|-------- Loaded data --------|" << std::endl;
		std::cout << "## restart file header says... ##" << std::endl;
		std::map<string, double>::iterator it;
		for (it = info.begin(); it != info.end(); it++) std::cout << it->first << " -> " << it->second << std::endl;
		std::cout << "##  end of restart file header ##" << std::endl;

		l.x = info["lx"];
		l.y = info["ly"];
		
		// TBD 
		nblocks = info["nblocks"];
		nthreads = info["nthreads"];

		nblocks = iDivUp(N, 128); //TODO
		nthreads = 128;
		
		// read strain related stuffs from restart data
		if ( info.find("num_of_strain_applied") != info.end() ) {
			std::cout << "Loading strain related data from restart file" << std::endl;
			is_strain_data_loaded = true;
  		
  		// if diff
  		if(strain_apply_interval != info["strain_apply_interval"]) {
  			std::cout << "strain_apply_interval from input        -> " << strain_apply_interval << std::endl;
  			std::cout << "strain_apply_interval from restart info -> " << info["strain_apply_interval"] << std::endl;
  			std::cout << "using values from input" << std::endl;
  		} else {
  			strain_apply_interval = info["strain_apply_interval"];
  		}

			if(strain_rate != info["strain_rate"]) {
  			std::cout << "shear_step from input        -> " << strain_rate << std::endl;
  			std::cout << "shear_step from restart info -> " << info["strain_rate"] << std::endl;
  			std::cout << "using values from input" << std::endl;
  		} else {
  			strain_rate = info["strain_rate"];
  		}
  		
  		if(e_bar != info["e_bar"]) {
  			std::cout << "e_bar from input        -> " << e_bar << std::endl;
  			std::cout << "e_bar from restart info -> " << info["e_bar"] << std::endl;
  			std::cout << "using values from input" << std::endl;
  		} else {
				e_bar = info["e_bar"];
			}
			
			num_of_strain_applied = 0; //info["num_of_strain_applied"];
			//epsilon_now = info["epsilon_now"];
			epsilon_initial = info["epsilon_now"];
			epsilon_now = epsilon_initial;
		}
		
		std::cout << strain_rate << " " << strain_apply_interval << " " << e_bar << std::endl;
		std::cout << num_of_strain_applied << "\t" << epsilon_now << std:: endl;
		
		break;
	}

	/************* LAUCH CONFIG ******************/
	fileIO->set_N(N);
	std::cout << "Grid config -> " << nblocks << " " << nthreads << std::endl;

	dev_allocate_thermo();
	copy_lattice_h2d();
	
	std::cout << "|------------------------------------------------------|" << std::endl;
	

	// *************** GENERATE LAMMPS DATA **************** //
	std::string lammps_str = generate_lammps_data(host_r, host_v, N, l.x, l.y);
	std::string lammps_file_name = prefix + ".lammps.dat";
	std::ofstream lammps_file(lammps_file_name.c_str());
	lammps_file << lammps_str;
	lammps_file.close();
	// ***************************************************** //

	//**************** REGISTER PER PARTICLE ARRAYS TO fileIO ************************ //
	fileIO->register_pp_qty("r", 0, host_r, 4);
	fileIO->register_pp_qty("v", 0, host_v, 4);
	fileIO->register_pp_qty("pid", 2, host_pid, 1);
	fileIO->register_pp_qty("stress_pp", 0, host_virial_tensor_pp, 4);

	//********************* declare thermos **************************************//
	double ke, pe, temperature, pressure;

	//NEW ONES
	double virial_pressure;
	double4 virial_tensor;
	double4 stress_tensor;
	double4 v2_tensor;

	// *************************** PRINT INPUTS **********************************//
	int ngpu = get_gpu_info();


	std::stringstream sinfo;

	sinfo << "# --------- BEGIN INPUT PARAMETERS -------- #" << std::endl;
	sinfo << "# Lattice ";
	if (lattice_init == NEW) {
		sinfo << " NEW " << std::endl;
		sinfo	<< "nx ny lx ly rho rc " << endl
			<< Nx << " " << Ny << " " << l.x << " " << l.y << " " << rho << " " << rc << std::endl;
	} else if (lattice_init == LOAD) {
		sinfo << " LOAD " << std::endl;
		sinfo	<< "n file_name" << std::endl
			<< N << " " << lattice_load_filename << std::endl;
	}

	sinfo << "# Integrator ";
	if (integrator_init == VELOCITY_VERLET) sinfo << "velocity_verlet" << std::endl;
	else if (integrator_init == LANGEVIN) sinfo << "LANGEVIN" << std::endl;
	sinfo << "tsteps_max dt " << std::endl;
	sinfo << tsteps_max << " " << dt << std::endl;

	sinfo << "# strain " << std::endl;
	sinfo << "strain_apply_interval strain_rate e_bar" << std::endl;
	sinfo << strain_apply_interval << " " << strain_rate << " " << e_bar << std::endl;

	sinfo << "# save" << std::endl;
	sinfo << "thermo_save_interval config_save_interval" << std::endl;
	sinfo << thermo_save_interval << " " << config_save_interval << std::endl;

	sinfo << "# ---------- BEGIN THERMO DATA ----------- #" << std::endl << std::endl;

	std::cout << sinfo.str();
	file_thermo << sinfo.str();

	// ****************************************************************************//

	cudaMemcpyToSymbol(dev_N, &N, sizeof(N));
	cudaMemcpyToSymbol(dev_l, &l, sizeof(l));
	cudaMemcpyToSymbol(dev_dt, &dt, sizeof(dt));


	// ************* init & calculate forces before time loop begins **************//
	set_accumulators_zero();

	if (is_NonBondedForces_initialized) {
		nonBondedForces->set_box_dim(l.x, l.y);
		nonBondedForces->compute();
	}

	if (is_BondedForces_initialized) {
		bondedForces->set_box_dim(l.x, l.y);
		bondedForces->init();
		bondedForces->compute();
	}

	if (is_NonAffine_inilialized) {
		nonAffine->set_box_dim(l.x, l.y);
		nonAffine->set_N_read_ref(N);
		nonAffine->calc_chi2(host_r);
	}
	
		//***************************************************************************//

	fileIO->add_pp_out_qty("pid", -1);
	fileIO->add_pp_out_qty("r", 0);
	fileIO->add_pp_out_qty("r", 1);
	fileIO->add_pp_out_qty("v", 0);
	fileIO->add_pp_out_qty("v", 1);
	fileIO->add_pp_out_qty("chi2", -1);
	fileIO->add_pp_out_qty("stress_pp", 0);
	fileIO->add_pp_out_qty("stress_pp", 1);
	fileIO->add_pp_out_qty("stress_pp", 2);
	fileIO->add_pp_out_qty("stress_pp", 3);

	//***** time records
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start);

	//calculate temperature before getting into time loop
	thermostat->apply();
	
	temperature = thermostat->get_temperature();
	ke = thermostat->get_ke();
	v2_tensor = thermostat->get_v_tensor();
	
	//thermo header
	file_thermo	<< std::setw(12) << "tstep" << " "
			<< std::setw(12) << "pe" << " "
			<< std::setw(12) << "ke" << " "
			<< std::setw(12) << "pressure" << " "
			<< std::setw(12) << "temperature" << " "
			<< std::setw(12) << "n_deformed" << " "
			<< std::setw(12) << "s_xx" << " "
			<< std::setw(12) << "s_xy" << " "
			<< std::setw(12) << "s_yx" << " "
			<< std::setw(12) << "s_yy" << " ";
			
	if (is_NonAffine_inilialized) {
		file_thermo << std::setw(12) << "[chi_mean]" << " "
					<< std::setw(12) << "[chi_fluc]" << " ";
	}

	file_thermo << std::setw(12) << "strain" << " ";
	file_thermo << std::endl;

	std::cout << thermo_save_interval << " " << config_save_interval << std::endl;
	std::stringstream sst;

	// calc chi2 < ADDED >
	//if (is_NonAffine_inilialized) nonAffine->calc_chi2(host_r);

	//----------------------------- TIME LOOP ----------------------------------//
	for (tstep = 0; tstep <= tsteps_max; tstep++) {
		/** ******************* PRINT THERMO *****************************************/
		if (tstep % thermo_save_interval == 0) {
			ke = thermostat->get_ke();
			pe = GlobalReduce(dev_pe_reduced, nblocks);
			virial_tensor = GlobalReduce(dev_virial_tensor_reduced, nblocks);
			virial_pressure = GlobalReduce(dev_virial_pressure_reduced, nblocks);
			v2_tensor = GlobalReduce(dev_v_tensor_reduced, nblocks);

			//do non affine stuffs
			//if (is_NonAffine_inilialized) nonAffine->calc_chi2(host_r);

			//compute thermos
			pressure = (N * temperature + 0.5 * virial_pressure) / (l.x * l.y);

			stress_tensor.x = (v2_tensor.x + virial_tensor.x) / (l.x * l.y);       //xx
			stress_tensor.y = (v2_tensor.y + virial_tensor.y) / (l.x * l.y);       //xy
			stress_tensor.z = (v2_tensor.z + virial_tensor.z) / (l.x * l.y);       //yx
			stress_tensor.w = (v2_tensor.w + virial_tensor.w) / (l.x * l.y);       //yy

			//progress
			std::cout << std::setw(12) << tstep << " " << num_of_strain_applied << "          \r";
			std::cout.flush();

			sst.str(" ");
			sst << setprecision(10);
			sst	<< std::setw(12) << tstep << " "
				<< std::setw(12) << pe / double(N) << " "
				<< std::setw(12) << ke / double(N) << " "
				<< std::setw(12) << pressure << " "
				<< std::setw(12) << temperature << " "
				<< std::setw(12) << num_of_strain_applied << " "
				<< std::setw(12) << stress_tensor.x << " "
				<< std::setw(12) << stress_tensor.y << " "
				<< std::setw(12) << stress_tensor.z << " "
				<< std::setw(12) << stress_tensor.w << " ";

			if (is_NonAffine_inilialized) {
				double chi_mean = nonAffine->get_mean_chi2();
				double chi_sqr_mean = nonAffine->get_mean_chi2_sqr();
				double chi_fluc = chi_sqr_mean - chi_mean*chi_mean;

				sst << std::setw(12) << chi_mean << " "
				    << std::setw(12) << chi_fluc;
			}

			sst	<< std::setw(12) << epsilon_now << " ";

			file_thermo << sst.str() << std::endl;
		}
	
		// ************************* PRINT LATTICE **************************** //
		if (tstep % config_save_interval == 0 && config_save_interval > 0) {
			std::cout << " [Writing lattice] \r";
			std::cout.flush();
			copy_lattice_d2h();
			sort_host_arrays_by_pid();
			//sort_dev_arrays_by_pid_and_copy();

			for (long i = 0; i < N; i++) {
				host_virial_tensor_pp[i].x = (host_v[i].x * host_v[i].x + host_virial_tensor_pp[i].x);  ///(l.x*l.y);
				host_virial_tensor_pp[i].y = (host_v[i].y * host_v[i].x + host_virial_tensor_pp[i].y);  ///(l.x*l.y);
				host_virial_tensor_pp[i].z = (host_v[i].x * host_v[i].y + host_virial_tensor_pp[i].z);  ///(l.x*l.y);
				host_virial_tensor_pp[i].w = (host_v[i].y * host_v[i].y + host_virial_tensor_pp[i].w);  ///(l.x*l.y);
			}

			//do non affine stuffs
			if (is_NonAffine_inilialized) nonAffine->get_pp_chi2();
			
			std::stringstream ss;
			ss << "# NEW_FRAME current_tstep lx ly N num_strain_applied strain_rate strain_apply_interval" << std::endl;
			ss << tstep << " " << l.x << " " <<  l.y << " " << N << " " << num_of_strain_applied << " " << strain_rate << " " << strain_apply_interval << std::endl;

			//fileIO->write_per_atom_qty(make_pp_header(tstep, tsteps_max));
			fileIO->write_per_atom_qty(ss.str());
		}

		/** ****************** MAIN TIME LOOP *********************************/ 
		// VV1 -> FORCE ->  **/

		if(integrator_init == VELOCITY_VERLET) preforce_velocity_verlet <<< nblocks, nthreads >>> (dev_r, dev_v, dev_f);
		CUDA_CHECK_RETURN(cudaThreadSynchronize());
		CUDA_CHECK_RETURN(cudaGetLastError());

		set_accumulators_zero();
		if (is_NonBondedForces_initialized) nonBondedForces->compute();
		if (is_BondedForces_initialized) bondedForces->compute();
		if (is_NonAffine_inilialized) nonAffine->calc_chi2(host_r);
		CUDA_CHECK_RETURN(cudaThreadSynchronize());
		CUDA_CHECK_RETURN(cudaGetLastError());

		if(integrator_init == VELOCITY_VERLET) postforce_velocity_verlet <<< nblocks, nthreads>>> (dev_v,dev_f);
		CUDA_CHECK_RETURN(cudaThreadSynchronize());
		CUDA_CHECK_RETURN(cudaGetLastError());

		// **************************** Thermostat **************************** //
		//calculate temperature here && apply thermostat if required
		thermostat->apply();
		temperature = thermostat->get_temperature();

		//if (is_NonAffine_inilialized) nonAffine->calc_chi2(host_r); // MOVED before postforce_vv

		// ********************* apply strain ******************************* //
		// STOP AT MAX_EPSILON
		double epsilon_next = epsilon_initial + 2.0*(num_of_strain_applied+1)*strain_rate;
		if(max_epsilon != -1 && epsilon_next>max_epsilon) {
		//if(max_epsilon != -1 && epsilon_now>max_epsilon) {
				//stop here
				std::cout << "epsilon_now/ max_epsilon = " << epsilon_now << " " << max_epsilon << std::endl;
				std::cout << "max_epsilon reached... stopping\nWish you a great luck!!" << std::endl;
				write_restart_data(N, prefix);
				return 0;
			}
				
		if (tstep > 0 && strain_apply_interval != 0 && tstep % strain_apply_interval == 0) {
			l.x += strain_rate * l.x;
			l.y -= strain_rate * l.y;
			cudaMemcpyToSymbol(dev_l, &l, sizeof(l));

			rescale_particle_pos <<< nblocks, nthreads >>> (dev_r, strain_rate);

			set_accumulators_zero();

			if (is_NonBondedForces_initialized) {
				nonBondedForces->set_box_dim(l.x, l.y);
				nonBondedForces->compute();
			}
			
			if (is_BondedForces_initialized) {
				bondedForces->set_box_dim(l.x,l.y);
				bondedForces->compute();
				
			}
	
			if (is_NonAffine_inilialized) {
				nonAffine->set_box_dim(l.x, l.y);
				nonAffine->scale_ref_lattice(strain_rate, -strain_rate); //
				nonAffine->calc_chi2(host_r);
			}

			num_of_strain_applied++;
			epsilon_now = epsilon_initial + 2.0*num_of_strain_applied*strain_rate;

			std::cout << tstep << " -> " << "# Box resized "
					<< l.x << " " << l.y << " " << "n_strain_app = " << num_of_strain_applied << " epsilon = " << epsilon_now << std::endl;
							
			//
			// STOP AT MAX_EPSILON WAS THERE
			//

		}
		
		//********************* RESTART DATA **************************//
		if(write_restart_data_interval != -1 && tstep > 0 && tstep%write_restart_data_interval == 0) {
				restart_data_count++;
				write_restart_data(N, std::string(prefix+"_"+boost::lexical_cast<std::string>(restart_data_count)));
			}
	}

	cudaEventRecord(stop);

	//**************** restart data ********************* //
	//copy_lattice_d2h();
	//sort_host_arrays_by_pid();
	//map<string, double> info_ = map_list_of("lx", l.x)("ly", l.y)("n", N)
	//				    ("nblocks", nblocks)("nthreads", nthreads)
	//				    ("tsteps_now", tstep)
	//				    ("tsteps_max", tsteps_max);

	write_restart_data(N, prefix);

	float ms = 0.0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("time loop took: %f ms\n", ms);


	return 0;
}
