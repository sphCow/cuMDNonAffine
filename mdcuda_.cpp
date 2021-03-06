////////////////////////////////////////////////////////////////////////////////////////////////
// Let it be a well-optimized GPU implementation of Molecular Dynamicss		  //
// in my one way. It might not be the fastest, might not be implemented 	  //
// in the most standard way, but It'd be "mine", my very own implementation   //
// of massively parallel MD in GPU. So, with love and passion towards physics //
// and computer programming, lets get started... 							  //
//																			  //
// Author -> Parswa Nath [TIFR-TCIS]										  //
// Use & redistribute as you wish, I don't care 							  //
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

#include "ThermostatBDP.h"

#include "nonaffine.h"
#include "BondedForces.h"
#include "NonBondedForces.h"

//kernels
//#include "kernels/map_pid2gtid.cu"

using namespace boost::assign;


#define dim 2;

// long N; // number of particles

// long  *host_pid;
// long  *host_cid;
// //double4 *host_r;
// double4 *host_v;
// double4 *host_f;

// long  *dev_pid;
// long  *dev_cid;
// double4 *dev_r;
// double4 *dev_v;
// double4 *dev_f;

//per atom stress
//double4 *dev_virial_tensor_pp;
double4 *host_virial_tensor_pp;

double rho;
double dt;

//rc & cell
double rc;
//uint3 ncell; //x,y,z=x*y
double3 l;
//double3 lcell;

//strain
double strain_rate;
int num_of_strain_applied = 0;

//Cell_List arrays
//uint2 *dev_cell_list; // x->start_id, y->len;
//uint4 *dev_cell_nebz1; // E NE N NW
//uint4 *dev_cell_nebz2; // W SW S SE
//int *dev_clen;
//int *dev_cbegin;

//uint2 *host_cell_list; // x->start_id, y->len;
//uint4 *host_cell_nebz1; // E NE N NW
//uint4 *host_cell_nebz2; // W SW S SE
//int *host_clen;
//int *host_cbegin;

// Kernel launch config
// long nblocks, nthreads;
int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

//****************** EXTERN PROTOTYPES ***************************//
//extern "C" void map_pid2gtid(long *dev_pid,long *dev_map_pid2gtid,const long &N ,const int &nblocks,const int &nthreads);

//****************************************************************//

// GPU info
int get_gpu_info() {
	int nDevices;

	printf("# ---------- GPU INFO ----------- #\n");

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
					 prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
					 prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
					 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
	}


	return nDevices;

}

// Generate lammps data
std::string generate_lammps_data(double4 *r, double4 *v, long N,double lx, double ly) {
    std::stringstream ss;

    ss << "LAMMPS Description" << std::endl;
    ss << std::endl;

    ss << N << " atoms" << std::endl;
    ss << "0 bonds\n0 angles\n0 dihedrals\n0 impropers" << std::endl;
    ss << std::endl;

    ss << "1 atom types" << std::endl<< std::endl;

    ss << std::setprecision(16) << -lx/2.0 << " " << std::setprecision(16) << lx/2.0 << " xlo xhi" << std::endl;
    ss << std::setprecision(16) << -ly/2.0 << " " << std::setprecision(16) << ly/2.0 << " ylo yhi" << std::endl;
    ss << "-0.5 0.5 zlo zhi" << std::endl << std::endl;

    ss << "Masses" << std::endl << std::endl;
    ss << "1 1.0" << std::endl << std::endl;

    ss << "Pair Coeffs" << std::endl << std::endl;
    ss << "1 1.0 1.0" << std::endl << std::endl;

    ss << "Atoms" << std::endl << std::endl;

    for(int i=0; i<N; i++) {
        ss << i+1 << " " << "1 " << std::setprecision(16) << r[i].x << " " << std::setprecision(16) << r[i].y << " 0.000" << std::endl;
    }

		ss << std::endl;
		ss << "Velocities" << std::endl << std::endl;

		for(int i=0; i<N; i++) {
				ss << i+1 << " " << std::setprecision(16) << v[i].x << " " << std::setprecision(16) << v[i].y << " 0.000" << std::endl;
		}

    return ss.str();
}

/////////////////////////////////////////////////////////////////////////////
//				Allocation/Data transfer/Print  			   //
// http://stackoverflow.com/questions/31598021/cuda-cudamemcpy-struct-of-arrays/31599025#comment51148602_31598021
/////////////////////////////////////////////////////////////////////////////
/***** Lattice *****/
void host_allocate_lattice(long N) {
	host_pid = new long[N]();
	host_cid = new long[N]();
	host_r = new double4[N]();
	host_v = new double4[N]();
	host_f = new double4[N]();
  host_virial_tensor_pp = new double4[N]();
  host_pid2gtid = new long[N]();
	printf("[HOST] allocated memory for lattice\n");
}

void dev_allocate_lattice(long N) {
	cudaMalloc((void **) &dev_pid, N*sizeof(*dev_pid));
	cudaMalloc((void **) &dev_cid, N*sizeof(long));
	cudaMalloc((void **) &dev_r, N*sizeof(*dev_r));
	cudaMalloc((void **) &dev_v, N*sizeof(*dev_v));
	cudaMalloc((void **) &dev_f, N*sizeof(*dev_f));

  cudaMalloc((void **) &dev_virial_tensor_pp, N*sizeof(*dev_virial_tensor_pp));

  cudaMalloc((void **) &dev_pid2gtid, N*sizeof(*dev_pid2gtid));

	printf("[DEVICE] allocated memory for lattice\n");
}

void copy_lattice_h2d() {
	cudaMemcpy(dev_pid, host_pid, N*sizeof(*dev_pid), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cid, host_cid, N*sizeof(*dev_cid), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r, host_r, N*sizeof(*dev_r), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_v, host_v, N*sizeof(*dev_v), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_f, host_f, N*sizeof(*dev_f), cudaMemcpyHostToDevice);
}

void copy_lattice_d2h() {
	cudaMemcpy(host_pid, dev_pid, N*sizeof(*host_pid), cudaMemcpyDeviceToHost);
	//cudaMemcpy(host_cid, dev_cid, N*sizeof(*host_cid), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_r, dev_r, N*sizeof(*host_r), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_v, dev_v, N*sizeof(*host_v), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_f, dev_f, N*sizeof(*host_f), cudaMemcpyDeviceToHost);
  cudaMemcpy(host_virial_tensor_pp, dev_virial_tensor_pp, N*sizeof(*host_virial_tensor_pp), cudaMemcpyDeviceToHost);

  //debug
  cudaMemcpy(host_pid2gtid, dev_pid2gtid, N*sizeof(*host_pid2gtid), cudaMemcpyDeviceToHost);


}


__host__ void sort_host_arrays_by_pid() {
  //sort by pid
	// thrust::sort_by_key(host_pid, host_pid+N,
	// 	thrust::make_zip_iterator(thrust::make_tuple(
	// 		host_cid, host_r, host_v, host_f, host_virial_tensor_pp)));

  thrust::sort_by_key(host_pid, host_pid+N,
		thrust::make_zip_iterator(thrust::make_tuple(
			host_r, host_v, host_virial_tensor_pp)));
}

std::string make_pp_header(long t, long tsteps_max) {
  std::stringstream ss;

  //make header TODO
  std::map<string,double> info = map_list_of ("current_tstep", t)
                                         ("lx", l.x)("ly", l.y)("n", N)
                                         ("nblocks",nblocks)("nthreads", nthreads)
                                         ("num_of_strain_applied", num_of_strain_applied)
                                         ("rc", rc)("strain_rate", strain_rate)
                                         ("tstep_now", t)
                                         ("tstep", tsteps_max);

  std::map<string, double>::iterator it;

  ss << "# ";
  for(it = info.begin(); it != info.end(); it++ ) ss << it->first << " ";
  ss << std::endl;

  ss << "# ";
  for(it = info.begin(); it != info.end(); it++ ) ss << std::setprecision(16) << double(it->second) << " ";
  ss << std::endl << std::endl;

  return ss.str();
}

void dev_allocate_thermo() {
	//scalars
	cudaMalloc((void **) &dev_thermo_pe, nblocks*sizeof(double));
	cudaMalloc((void **) &dev_thermo_v2, nblocks*sizeof(double));
	cudaMalloc((void **) &dev_thermo_virial, nblocks*sizeof(double));

	//vectors (matrices, xx, xy, yx, yy)
	cudaMalloc((void **) &dev_thermo_v_ij, nblocks*sizeof(double4));

	//virial tensor
	cudaMalloc((void **) &dev_thermo_virial_xx, nblocks*sizeof(double));
	cudaMalloc((void **) &dev_thermo_virial_xy, nblocks*sizeof(double));
	cudaMalloc((void **) &dev_thermo_virial_yy, nblocks*sizeof(double));

	// KE tensor
	cudaMalloc((void **) &dev_thermo_v_xx, nblocks*sizeof(double));
	cudaMalloc((void **) &dev_thermo_v_xy, nblocks*sizeof(double));
	cudaMalloc((void **) &dev_thermo_v_yy, nblocks*sizeof(double));

	// Born
	cudaMalloc((void **) &dev_thermo_born_xx, nblocks*sizeof(double));
	cudaMalloc((void **) &dev_thermo_born_xy, nblocks*sizeof(double));
	cudaMalloc((void **) &dev_thermo_born_yy, nblocks*sizeof(double));
	
	// *********** NEW ONES **************//
	//double* dev_pe_reduced = 0;
	//double4* dev_virial_tensor_pp = 0;
	//double4* dev_virial_tensor_reduced = 0;
	//double* dev_virial_pressure_reduced = 0;
	
	cudaMalloc((void **) &dev_pe_reduced, nblocks*sizeof(double));
	cudaMalloc((void **) &dev_virial_tensor_reduced, nblocks*sizeof(double4));
	cudaMalloc((void **) &dev_virial_pressure_reduced, nblocks*sizeof(double));
	cudaMalloc((void **) &dev_virial_tensor_pp, N*sizeof(double4));

}



/////////////////////////////////////////////////////////////////////////////
// [CPU] Generate Lattice									   //
// sets l.x,ly
// TODO : sample velocity from Gaussian dist.                  //
/////////////////////////////////////////////////////////////////////////////
__host__ void generate_lattice(long Nx, long Ny, int type) {
	double ax,ay;

	if(type==0) {
        printf("# generating Triangular lattice %ld %ld %lf \n", Nx, Ny, rho);
        ax = sqrt(2.0 / (rho * sqrt(3.0)));
        ay = ax * sqrt(3.0) / 2.0;
    }

    else if(type==1) {
        printf("# generating square lattice\n");
        ax = sqrt(0.0/rho);
        ay = ax;
    }

    else {
        fprintf(stderr,"unknown lattice!\n");
    }

    l.x = Nx * ax;
    l.y = Ny * ay;

    double lxh = l.x / 2.0;
    double lyh = l.y / 2.0;

    srand48(100);

    for(long i = 0; i < Nx; i++) {
        for(long j = 0; j < Ny; j++) {
            double xx = -lxh + i * ax;
            double yy = -lyh + j * ay;

            if(type==0 && j % 2 != 0)
                xx += ax / 2.0;

            long id = i + j * Nx;

        	host_pid[id] = id;
            host_r[id].x = xx; //x
            host_r[id].y = yy; //y
            host_r[id].z = 0.0; //phi //TODO

            host_v[id].x = drand48()-0.5; //x //TODO
            host_v[id].y = drand48()-0.5; //y //TODO
            host_v[id].z = 0.0; //phi //TODO

            host_f[id].x = 0.0; //x
            host_f[id].y = 0.0; //y
            host_f[id].z = 0.0; //phi
         }
    }
}

__host__ void init_velocities(double target_T) {
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
    double temp = 0.5 * sumv2 / double(N);
    double fac = sqrt(target_T / temp);

    for (unsigned int i = 0; i < N; i++) {
        host_v[i].x *= fac;
        host_v[i].y *= fac;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Reduce over threads kernel                                                 //
//
////////////////////////////////////////////////////////////////////////////////
__device__ void reduce_over_thread(volatile double* sdata, double *g_odata) {
	unsigned int tid = threadIdx.x;
	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    //__syncthreads();
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__global__ void preforce_velocity_verlet(double4 *r_, double4 *v_, double4 *f_) {
	int gtid = blockDim.x*blockIdx.x + threadIdx.x;
	if(gtid >= dev_N) return;

	if(gtid==1000) printf("preforce %d %lf %lf %lf\n",dev_N,dev_l.x,dev_l.y,dev_dt);

	//read to registers
	double4 r = r_[gtid];
	double4 v = v_[gtid];
	double4 f = f_[gtid];
	double dt = dev_dt;

	//full update r
	r.x += dt*v.x + 0.5*dt*dt*f.x;
	r.y += dt*v.y + 0.5*dt*dt*f.y;

	//pbc
	r.x -= dev_l.x * rint(r.x/dev_l.x);
	r.y -= dev_l.y * rint(r.y/dev_l.y);

	//DEBUG
	//uint my_cx = uint((p->r[gtid].x + dev_l.x/2.0) / dev_lcell.x);
    //uint my_cy = uint((p->r[gtid].y + dev_l.y/2.0) / dev_lcell.y);
    //uint my_cid = (my_cx) + (my_cy) * (dev_ncell.x);
    //p->cid[gtid] = my_cid;

	//half update v
	v.x += 0.5*dt*f.x;
	v.y += 0.5*dt*f.y;

	//write r & half updated v to global mem
	r_[gtid] = r;
	v_[gtid] = v;
}


__global__ void postforce_velocity_verlet(double4* v_,double4* f_ ,double *dev_thermo_v2) {
	extern __shared__ double shared_thermo[];

	int gtid = blockDim.x*blockIdx.x + threadIdx.x;
	if(gtid >= dev_N) return;

	if(gtid==1000) printf("postforce %d %lf %lf %lf\n",dev_N,dev_l.x,dev_l.y,dev_dt);

	//read to registers
	double4 v = v_[gtid];
	double4 f = f_[gtid];
	double dt = dev_dt;

	//full update velocity
	v.x += 0.5*dt*f.x;
	v.y += 0.5*dt*f.y;

	//write to global mem
	v_[gtid] = v;

	//thermo - sum over v**2
	shared_thermo[threadIdx.x] = v.x*v.x + v.y*v.y;
	__syncthreads();
	reduce_over_thread(shared_thermo, dev_thermo_v2);

	// //thermo - SUM[v_xy]_i
	// shared_thermo[threadIdx.x] = v.x*v.x;
	// __syncthreads();
	// reduce_over_thread(shared_thermo, dev_thermo_v_xx);
	//
	// shared_thermo[threadIdx.x] =v.y*v.y;
	// __syncthreads();
	// reduce_over_thread(shared_thermo, dev_thermo_v_yy);
	//
	// shared_thermo[threadIdx.x] = v.x*v.y;
	// __syncthreads();
	// reduce_over_thread(shared_thermo, dev_thermo_v_xy);


}

__global__ void rescale_velocities(double4* v_,double scale_factor,
											 double *dev_thermo_v2,
											 double *dev_thermo_v_xx,
											 double *dev_thermo_v_yy,
											 double *dev_thermo_v_xy) {
	extern __shared__ double shared_thermo[];

	int gtid = blockDim.x*blockIdx.x + threadIdx.x;
	if(gtid >= dev_N) return;

	//read to registers
	double4 v = v_[gtid];
	v.x*=scale_factor;
	v.y*=scale_factor;

	//write back to global mem
	v_[gtid] = v;

	//thermo - temperature
	shared_thermo[threadIdx.x] = v.x*v.x + v.y*v.y;
	__syncthreads();
	reduce_over_thread(shared_thermo, dev_thermo_v2);

	//thermo - SUM[v_xy]_i
	shared_thermo[threadIdx.x] = v.x*v.x;
	__syncthreads();
	reduce_over_thread(shared_thermo, dev_thermo_v_xx);

	shared_thermo[threadIdx.x] =v.y*v.y;
	__syncthreads();
	reduce_over_thread(shared_thermo, dev_thermo_v_yy);

	shared_thermo[threadIdx.x] = v.x*v.y;
	__syncthreads();
	reduce_over_thread(shared_thermo, dev_thermo_v_xy);

}

__global__ void rescale_particle_pos(double4* r_,double scale_factor) {

	int gtid = blockDim.x*blockIdx.x + threadIdx.x;
	if(gtid >= dev_N) return;

	//read to registers
	double4 r = r_[gtid];
	r.x=(1.0 + scale_factor)*r.x;
	r.y=(1.0 - scale_factor)*r.y;

	//write back to global mem
	r_[gtid] = r;
}

void write_restart_data(long N,std::string file_name,std::map<string, double> &info) {
    ofstream f;
		std::string name = file_name+".restart.out";
		f.open(name.c_str());

    f << "Begin_Info" << std::endl;

		std::map<string, double>::iterator it;

    for(it = info.begin(); it != info.end(); it++ ) f << it->first << " ";
    f << std::endl;

		for(it = info.begin(); it != info.end(); it++ ) f << std::setprecision(16) << it->second << " ";
    f << std::endl;

    // vector<string> header = {"id", "Pe", "x", "y","z",
    //                          "vx","vy","vz",
    //                          "fx", "fy", "fz"};

    f << std::endl << "Begin_Particle_Data" << std::endl;

    for(long i=0; i<N; i++) {
        f << host_pid[i] << " "
          << host_r[i].x << " " << host_r[i].y <<" "<< host_r[i].z << " "
          << host_v[i].x << " " << host_v[i].y <<" "<< host_v[i].z << std::endl;
    }

    std::cout << "restart data written to " << file_name << ".restart.out" << std::endl;

    f.close();
}

int load_lattice(std::map<string,double> &info, string filename) {
//int load_lattice(long &N, long  *host_pid, double4 *host_r, double4 *host_v, std::map<string,double> &info, string filename) {
	ifstream f;
  f.open(filename.c_str());

  if(!f.good()) return -1;

  string line;
  long lcount = 0;
  long header_begin = -1;
  long lattice_begin = -1;
	long particle_count = 0;

  std::vector<string> header_key;
  std::vector<string> header_val;

  while(getline(f,line)) {

      if(line=="Begin_Info") {
          header_begin = lcount;
      }

      else if(line=="Begin_Particle_Data") {
          lattice_begin = lcount+1;

					for(int i=0; i<header_key.size(); i++) {
				      // info.emplace(header_key[i], boost::lexical_cast<double>(header_val[i]));
							info[header_key[i]] = boost::lexical_cast<double>(header_val[i]);
				  }

					// ***** ALLOCATION HERE ******//
					N = boost::lexical_cast<long>(info["n"]);
					std:: cout << "from load " << N << std::endl;
					host_allocate_lattice(N);
					dev_allocate_lattice(N);
      }

      if(lcount == (header_begin+1)) {
          stringstream iss(line);
          copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(header_key));
      }

      if(lcount == (header_begin+2)) {
          stringstream iss(line);
          copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(header_val));
      }

      // LATTICE
      if(lattice_begin != -1 && lcount >= lattice_begin) {
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
  if(N!=long(info["n"])) {
      cerr << "Number of particles doesn't match" << endl;
      return -1;
  }

    return 0;
}

int main(int argc, char** argv) {

  FileIO *fileIO = new FileIO;

  int gpuid = 0;
  if(argc==2) {
    gpuid = boost::lexical_cast<int>(argv[1]);
  }

  cudaSetDevice(gpuid);
  std::cout << "CUDA Device set to GPU-" << gpuid << std::endl;


  //FileIO* fileIO;

	enum lattice_init_t {LATTICE_NOT_DEFINED, NEW, LOAD};
	enum thermostat_init_t {THERMOSTAT_NOT_DEFINED, BDP, BERENSDEN, NO};
	enum integrator_init_t {INTEGRATOR_NOT_DEFINED, VELOCITY_VERLET, LANGEVIN, NONE};

	lattice_init_t lattice_init = LATTICE_NOT_DEFINED;
	thermostat_init_t thermostat_init = THERMOSTAT_NOT_DEFINED;
	integrator_init_t integrator_init = INTEGRATOR_NOT_DEFINED;

	long Nx, Ny;
	unsigned long tsteps_max;

  /***** classes *************/
  NonAffine *nonAffine;
  bool is_NonAffine_inilialized = false;

  BondedForces *bondedForces;
  bool is_BondedForces_initialized = false;
  
  NonBondedForces *nonBondedForces;
  bool is_NonBondedForces_initialized = false;

	/***** Interpret inputs ******/
	double target_termperature; // = 1.0;
	double tau_t; // = 0.1;
	unsigned long strain_apply_interval;// = 0;
	unsigned long thermo_save_interval; //= 100;
	unsigned long config_save_interval; //= 0;
	std::string prefix; //= "just_another_md";
	std::string lattice_load_filename;
	int lj_smoothing_level;

	std::string line;
	while(std::getline(cin, line)) {
		std::vector<std::string> tokenized;
	//	if(!line.empty() && line[0]!='#' )
    	if(line[0]=='#' || line.empty()) continue; //skip empty lines && lines starting with #
			fileIO->tokenize(line, tokenized);

			// prefix
			if(tokenized[0] == "prefix") {
				prefix = tokenized[1];
				std::cout << "prefix set to -> " << prefix << std::endl;
      			fileIO = new FileIO;
      			fileIO->set_file_prefix(prefix);
			}

			else if(tokenized[0] == "non_bonded_force") {
				nonBondedForces = new NonBondedForces;
				is_NonBondedForces_initialized = true;
				nonBondedForces->interpret(tokenized);
			}

			// lattice
			else if(tokenized[0] == "lattice" && lattice_init == LATTICE_NOT_DEFINED) {
				if(tokenized[1] == "new") {
					std::cout << "new lattice to be generated \n";
					Nx = boost::lexical_cast<long>(tokenized[3]);
					Ny = boost::lexical_cast<long>(tokenized[4]);
					N = Nx*Ny;
					rho = boost::lexical_cast<double>(tokenized[5]);
					lattice_init = NEW;
				}

				else if(tokenized[1] == "load") {
					std::cout << "lattice to be loaded from " << tokenized[2] << std::endl;
					lattice_init = LOAD;
					lattice_load_filename = tokenized[2];
				}

				else {
					std::cerr << "Unknown lattice args" << std::endl;
				}


			}

			//integrator
			else if(tokenized[0] == "integrator" && integrator_init == INTEGRATOR_NOT_DEFINED) {
				if(tokenized[1]=="velocity_verlet") {
					std::cout << "integrator -> " << "velocity_verlet" << std::endl;
					integrator_init = VELOCITY_VERLET;
					dt = boost::lexical_cast<double>(tokenized[2]);
					tsteps_max = boost::lexical_cast<long>(tokenized[3]);
				}

				else if(tokenized[1]=="langevin") {
					integrator_init = LANGEVIN;
					std::cout << "integrator -> " << "langevin" << std::endl;
				}

      			else if(tokenized[1]=="none") {
					integrator_init = NONE;
					std::cout << "integrator -> " << "NO INTERGRATOR" << std::endl;
				}

			else std::cerr << "unknown integrator" << std::endl;

			}

			//strain
			else if(tokenized[0] == "strain") {
				strain_apply_interval = boost::lexical_cast<long>(tokenized[1]);
				strain_rate = boost::lexical_cast<double>(tokenized[2]);
			}

			// thermostat
			else if(tokenized[0] == "thermostat") {
				if(tokenized[1] == "BDP") {
					thermostat_init = BDP;
					target_termperature = boost::lexical_cast<double>(tokenized[2]);
					tau_t = boost::lexical_cast<double>(tokenized[3]);
				}

				else if(tokenized[1] == "berensden") {
					thermostat_init = BERENSDEN;
					target_termperature = boost::lexical_cast<double>(tokenized[2]);
					tau_t = boost::lexical_cast<double>(tokenized[3]);
				}

				else {
					std::cerr << "unrecognized thermostat" << std::endl;
				}


		}

    	//NonAffine
    	else if(tokenized[0] == "non_affine") {
      		nonAffine = new NonAffine;
      		is_NonAffine_inilialized = true;
      		nonAffine->interpret(tokenized);
    	}

    	//BondedForces
    	else if(tokenized[0] == "bonded_force") {
      		bondedForces = new BondedForces;
      		is_BondedForces_initialized = true;
      		bondedForces->interpret(tokenized);
    	}

		//save
		else if(tokenized[0] == "save") {
			thermo_save_interval = boost::lexical_cast<long>(tokenized[1]);
			config_save_interval = boost::lexical_cast<long>(tokenized[2]);
		}

		else {
			std::cout << "ignoring unrecognized/duplicate keyword(s) " << tokenized[0] << endl;
		}

	}

	/**** Thermostat **********/

	ThermostatBDP tstat;
	tstat.set(tau_t/dt);

	/********** Files ***********/
	std::ofstream file_thermo;
	//std::ofstream file_lattice;
	std::ofstream file_log;

	std::string file_thermo_name = prefix+".thermo";
	//std::string file_lattice_name = prefix+".lattice";
	std::string file_log_name = prefix+".log";

	file_thermo.open(file_thermo_name.c_str());
	file_log.open(file_log_name.c_str());
	//if(config_save_interval>0)
  //file_lattice.open(file_lattice_name.c_str());

	// ****************** generate or load lattice ***************** //
	std::map<std::string,double> info;

	switch(lattice_init) {
		case NEW:
			host_allocate_lattice(N);
			dev_allocate_lattice(N);
			generate_lattice(Nx,Ny,0);
			init_velocities(target_termperature);

      //nblocks = Nx;
			//nthreads = Ny;

      		nblocks = iDivUp(N, 128); //TODO
      		nthreads = 128;
			break;

    case LOAD:
			std::cout << "case load " << std::endl;
			int load_result = load_lattice(info, lattice_load_filename);

			if(load_result!=0) {
				std::cerr << "can't load lattice" << std::endl;
				return -1;
			}

			//if loaded successfully DEBUG
      		std::cout << " ** Loaded data **" << std::endl;
			std::map<string,double>::iterator it;
			for(it=info.begin();it!=info.end();it++) std::cout << it->first << " " << it->second << std::endl;
      		std::cout << "** **" << std::endl;

			l.x = info["lx"];
			l.y = info["ly"];
			nblocks = info["nblocks"];
			nthreads = info["nthreads"];

      		nblocks = iDivUp(N, 128); //TODO
      		nthreads = 128;

			//printf("N -> %ld\n", N);
			printf("[DEVICE] Grid config -> %ld x %ld\n", nblocks, nthreads);

			break;
	}

  /************* LAUCH CONFIG ******************/
  fileIO->set_N(N);
  std::cout << "N -> " << N <<std::endl;
  std::cout << "Grid config -> " << nblocks << " " << nthreads << std::endl;

	dev_allocate_thermo();
	copy_lattice_h2d();

	// *************** GENERATE LAMMPS DATA **************** //
	std::string lammps_str = generate_lammps_data(host_r, host_v, N, l.x,l.y);
	std::string lammps_file_name = prefix+".lammps.dat";
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
	double ke,pe,sum_v2,temperature,pressure;
	double s_xx, s_xy, s_yy;
	double v_xx, v_xy, v_yy;
	double virial_sum, virial_xx, virial_xy, virial_yy;
	double born_xx, born_xy, born_yy;

	// *************************** PRINT INPUTS **********************************//
	int ngpu = get_gpu_info();


	std::stringstream sinfo;

	sinfo << "# --------- BEGIN INPUT PARAMETERS -------- #" << std::endl;
	sinfo << "# Lattice ";
	if(lattice_init == NEW) {
		sinfo << " NEW " << std::endl;
		sinfo << "nx ny lx ly rho rc " << endl
					<< Nx << " " << Ny << " " << l.x << " " << l.y << " " << rho << " " << rc << std::endl;
	}

	else if(lattice_init == LOAD) {
		sinfo << " LOAD " << std::endl;
		sinfo << "n file_name" << std::endl
					<< N << " " << lattice_load_filename << std::endl;
	}

	sinfo << "# Thermostat ";
	if(thermostat_init == BDP) sinfo << "BDP" << std::endl;
	else if(thermostat_init == BERENSDEN) sinfo << "Berensden" << std::endl;
	sinfo << "target_temp taut " << std::endl;
	sinfo << target_termperature << " " << tau_t << std::endl;

	sinfo << "# Integrator ";
	if(integrator_init == VELOCITY_VERLET) sinfo << "velocity_verlet" << std::endl;
	else if(integrator_init == LANGEVIN) sinfo << "LANGEVIN" << std::endl;
	sinfo << "tsteps_max dt " << std::endl;
	sinfo << tsteps_max << " " << dt << std::endl;

	sinfo << "# strain " << std::endl;
	sinfo << "strain_apply_interval strain_rate" << std::endl;
	sinfo << strain_apply_interval << " " << strain_rate << std::endl;

	sinfo << "# save" << std::endl;
	sinfo << "thermo_save_interval config_save_interval" << std::endl;
	sinfo << thermo_save_interval << " " << config_save_interval << std::endl;

	sinfo << "# ---------- BEGIN THERMO DATA ----------- #" << std::endl << std::endl;

	std::cout << sinfo.str();
	file_thermo << sinfo.str();
	//***************************************************************************//

  // ****** initialize NONAFFINE stuffs ****************//
  if(is_NonAffine_inilialized) {
    nonAffine->set_box_dim(l.x,l.y);
    nonAffine->set_N_read_ref(N);
    //nonAffine->calc_chi2(host_r);
  }

  fileIO->add_pp_out_qty("pid", -1);
  fileIO->add_pp_out_qty("r",0);
  fileIO->add_pp_out_qty("r",1);
  fileIO->add_pp_out_qty("chi2",-1);
  fileIO->add_pp_out_qty("stress_pp",0);
  fileIO->add_pp_out_qty("stress_pp",1);
  fileIO->add_pp_out_qty("stress_pp",2);
  fileIO->add_pp_out_qty("stress_pp",3);

  // ****************************************************************************//

	cudaMemcpyToSymbol(dev_l, &l, sizeof(l));
	cudaMemcpyToSymbol(dev_dt, &dt, sizeof(dt));

		if(is_NonBondedForces_initialized) {
    	nonBondedForces->set_box_dim(l.x,l.y);
  	}		

  if(is_BondedForces_initialized) {
    bondedForces->init();
  }


	//***** time records
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start);

	//*** thrust pointers
	thrust::device_ptr<double> dev_ptr_thermo_pe = thrust::device_pointer_cast(dev_thermo_pe);
	thrust::device_ptr<double> dev_ptr_thermo_virial = thrust::device_pointer_cast(dev_thermo_virial);
	// thrust::device_ptr<double> dev_ptr_thermo_v2 = thrust::device_pointer_cast(dev_thermo_v2);
	//virial_xy
	thrust::device_ptr<double> dev_ptr_thermo_virial_xx = thrust::device_pointer_cast(dev_thermo_virial_xx);
	thrust::device_ptr<double> dev_ptr_thermo_virial_xy = thrust::device_pointer_cast(dev_thermo_virial_xy);
	thrust::device_ptr<double> dev_ptr_thermo_virial_yy = thrust::device_pointer_cast(dev_thermo_virial_yy);
	// v_xy
	thrust::device_ptr<double> dev_ptr_thermo_v_xx = thrust::device_pointer_cast(dev_thermo_v_xx);
	thrust::device_ptr<double> dev_ptr_thermo_v_yy = thrust::device_pointer_cast(dev_thermo_v_yy);
	thrust::device_ptr<double> dev_ptr_thermo_v_xy = thrust::device_pointer_cast(dev_thermo_v_xy);
	//born
	thrust::device_ptr<double> dev_ptr_thermo_born_xx = thrust::device_pointer_cast(dev_thermo_born_xx);
	thrust::device_ptr<double> dev_ptr_thermo_born_xy = thrust::device_pointer_cast(dev_thermo_born_xy);
	thrust::device_ptr<double> dev_ptr_thermo_born_yy = thrust::device_pointer_cast(dev_thermo_born_yy);

	if(is_NonBondedForces_initialized) nonBondedForces->compute();
  	if(is_BondedForces_initialized) bondedForces->calculate_non_bonded_force();
	if(is_NonAffine_inilialized) nonAffine->calc_chi2(host_r);
  

	//calculate temperature before getting into time loop
	rescale_velocities<<<nblocks,nthreads,nthreads*sizeof(double)>>>(dev_v, 1.0,
																   dev_thermo_v2,
																   dev_thermo_v_xx,
																   dev_thermo_v_yy,
																   dev_thermo_v_xy);
  	thrust::device_ptr<double> dev_ptr_thermo_v2;// = thrust::device_pointer_cast(dev_thermo_v2);
  	dev_ptr_thermo_v2 = thrust::device_pointer_cast(dev_thermo_v2);
  	sum_v2 = thrust::reduce(dev_ptr_thermo_v2, dev_ptr_thermo_v2 + nblocks); //
	temperature = sum_v2/double(2*N);
	ke = 0.5*sum_v2;
	v_xx = thrust::reduce(dev_ptr_thermo_v_xx, dev_ptr_thermo_v_xx + nblocks);
	v_xy = thrust::reduce(dev_ptr_thermo_v_xy, dev_ptr_thermo_v_xy + nblocks);
	v_yy = thrust::reduce(dev_ptr_thermo_v_yy, dev_ptr_thermo_v_yy + nblocks);


	//thermo header
	file_thermo << std::setw(12) << "tstep" << " "
							<< std::setw(12) << "pe" << " "
							<< std::setw(12) << "ke" << " "
							<< std::setw(12) << "pressure" << " "
							<< std::setw(12) << "temperature" << " "
              << std::setw(12) << "n_deformed" << " "
							<< std::setw(12) << "s_xx" << " "
							<< std::setw(12) << "s_xy" << " "
							<< std::setw(12) << "s_yy" << " "
							<< std::setw(12) << "born_xx" << " "
							<< std::setw(12) << "born_xy" << " "
							<< std::setw(12) << "born_yy" << " ";

  if(is_NonAffine_inilialized) file_thermo << " mean_chi2" ;
	file_thermo << std::endl;

  std::cout << thermo_save_interval << " " << config_save_interval << std::endl;
  std::stringstream sst;

	//----------------------------- TIME LOOP ----------------------------------//
	for(tstep = 0; tstep<tsteps_max; tstep++) {

    // ************************* PRINT LATTICE **************************** //
    	if(tstep%config_save_interval==0 && config_save_interval>0) {
      		std::cout << std::setw(12) << tstep << " Writing lattice\n";
      		copy_lattice_d2h();
      		sort_host_arrays_by_pid();

      		for(long i=0; i<N; i++) {
        		host_virial_tensor_pp[i].x = (host_v[i].x*host_v[i].x + host_virial_tensor_pp[i].x);///(l.x*l.y);
        		host_virial_tensor_pp[i].y = (host_v[i].y*host_v[i].x + host_virial_tensor_pp[i].y);///(l.x*l.y);
        		host_virial_tensor_pp[i].z = (host_v[i].x*host_v[i].y + host_virial_tensor_pp[i].z);///(l.x*l.y);
        		host_virial_tensor_pp[i].w = (host_v[i].y*host_v[i].y + host_virial_tensor_pp[i].w);///(l.x*l.y);
      		}

	      	//do non affine stuffs
    	  	if(is_NonAffine_inilialized) nonAffine->get_pp_chi2();
      
      		fileIO->write_per_atom_qty(make_pp_header(tstep, tsteps_max));
    	}

		/** ******************* PRINT THERMO *****************************************/
		if(tstep%thermo_save_interval==0) {
			pe = thrust::reduce(dev_ptr_thermo_pe, dev_ptr_thermo_pe+nblocks);
			virial_sum = thrust::reduce(dev_ptr_thermo_virial, dev_ptr_thermo_virial+nblocks);

			//virial tensor
			virial_xx = thrust::reduce(dev_ptr_thermo_virial_xx, dev_ptr_thermo_virial_xx + nblocks);
			virial_xy = thrust::reduce(dev_ptr_thermo_virial_xy, dev_ptr_thermo_virial_xy + nblocks);
			virial_yy = thrust::reduce(dev_ptr_thermo_virial_yy, dev_ptr_thermo_virial_yy + nblocks);

			// v_xy
			v_xx = thrust::reduce(dev_ptr_thermo_v_xx, dev_ptr_thermo_v_xx + nblocks);
			v_xy = thrust::reduce(dev_ptr_thermo_v_xy, dev_ptr_thermo_v_xy + nblocks);
			v_yy = thrust::reduce(dev_ptr_thermo_v_yy, dev_ptr_thermo_v_yy + nblocks);

			//born
			born_xx = thrust::reduce(dev_ptr_thermo_born_xx, dev_ptr_thermo_born_xx + nblocks)/(l.x*l.y);
			born_xy = thrust::reduce(dev_ptr_thermo_born_xy, dev_ptr_thermo_born_xy + nblocks)/(l.x*l.y);
			born_yy = thrust::reduce(dev_ptr_thermo_born_yy, dev_ptr_thermo_born_yy + nblocks)/(l.x*l.y);

			//compute thermos
			pressure = (N*temperature + 0.5*virial_sum)/(l.x*l.y);
			s_xx = (v_xx + virial_xx)/(l.x*l.y);
			s_yy = (v_yy + virial_yy)/(l.x*l.y);
			s_xy = (v_xy + virial_xy)/(l.x*l.y);

      		//progress
      		std::cout << std::setw(12) << tstep << " " << num_of_strain_applied << "\n" ;

      		sst.str(" ");
      		sst << std::setw(12) << tstep << " "
									<< std::setw(12) << pe/double(N) << " "
									<< std::setw(12) << ke/double(N) << " "
									<< std::setw(12) << pressure << " "
									<< std::setw(12) << temperature << " "
                  					<< std::setw(12) << num_of_strain_applied << " "
									<< std::setw(12) << s_xx << " "
									<< std::setw(12) << s_xy << " "
									<< std::setw(12) << s_yy  << " "
									<< std::setw(12) << born_xx << " "
									<< std::setw(12) << born_xy << " "
									<< std::setw(12) << born_yy;

      		if(is_NonAffine_inilialized) sst << " " << nonAffine->get_mean_chi2();

      		file_thermo << sst.str() << std::endl;

		}


		/** ******************END_PRINT **/

		preforce_velocity_verlet<<<nblocks,nthreads>>>(dev_r, dev_v, dev_f);
		CUDA_CHECK_RETURN(cudaThreadSynchronize());
		CUDA_CHECK_RETURN(cudaGetLastError());

		if(is_NonBondedForces_initialized) nonBondedForces->compute();	
    	if(is_BondedForces_initialized) bondedForces->calculate_non_bonded_force();
    	
    	//debug 
    	cudaMemcpy(host_f, dev_f, N*sizeof(*host_f), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_pid, dev_pid, N*sizeof(*host_pid), cudaMemcpyDeviceToHost);
	std::cerr << host_pid[100] << " " << host_f[100].x << " " << host_f[100].y << std::endl;
    	
    	
		postforce_velocity_verlet<<<nblocks,nthreads,nthreads*sizeof(double)>>>(dev_v,
																			  dev_f,
																			  dev_thermo_v2);
		CUDA_CHECK_RETURN(cudaThreadSynchronize());
		CUDA_CHECK_RETURN(cudaGetLastError());

		// **************************** Thermostat **************************** //
		//calculate temperature here && apply thermostat if required
		dev_ptr_thermo_v2 = thrust::device_pointer_cast(dev_thermo_v2);
		sum_v2 = thrust::reduce(dev_ptr_thermo_v2, dev_ptr_thermo_v2 + nblocks); //
		temperature = sum_v2/double(2*N);
		ke=0.5*sum_v2;

		double new_ke = tstat.resamplekin(ke, N*target_termperature, 2*N);
		double lambda = std::sqrt(new_ke/ke);

		//rescale and recalculate ke,temp,v_ij
		rescale_velocities<<<nblocks,nthreads,nthreads*sizeof(double)>>>(dev_v,
																	   lambda,
																	   dev_thermo_v2,
																	   dev_thermo_v_xx,
																	   dev_thermo_v_yy,
																	   dev_thermo_v_xy);

		dev_ptr_thermo_v2 = thrust::device_pointer_cast(dev_thermo_v2);
		sum_v2 = thrust::reduce(dev_ptr_thermo_v2, dev_ptr_thermo_v2 + nblocks); //
		temperature = sum_v2/double(2*N);
		ke=0.5*sum_v2;

		std::cout << temperature << std::endl;

    	//do non affine stuffs
    	if(is_NonAffine_inilialized) nonAffine->calc_chi2(host_r);
    	
		//debug 
		//copy_lattice_d2h();
		//for(int i=0; i<N; i++) file_log << host_pid[i] << " " << host_f[i].x << " " << host_f[i].y << std::endl;


		// ********************* apply strain ******************************* //
		if(tstep>0 && strain_apply_interval!=0 && tstep%strain_apply_interval==0) {
			l.x += strain_rate*l.x;
			l.y -= strain_rate*l.y;
			cudaMemcpyToSymbol(dev_l, &l, sizeof(l));
			
			if(is_NonBondedForces_initialized) nonBondedForces->set_box_dim(l.x,l.y);

      		if(is_NonAffine_inilialized) {
        		nonAffine->set_box_dim(l.x,l.y);
        		nonAffine->scale_ref_lattice(strain_rate, -strain_rate); //
      		}

      		rescale_particle_pos<<<nblocks,nthreads>>>(dev_r,strain_rate);
      	
			num_of_strain_applied++;

			std::cout << tstep << " -> " << "# Box resized "
               << l.x << " " << l.y << " " << l.x*l.y << " "
               << num_of_strain_applied << std::endl;

      		file_log << tstep << " " << num_of_strain_applied << " "
              << l.x << " " << l.y << " " << l.x*l.y << " "
              << std::endl;

		  //***********
			if(is_NonBondedForces_initialized) nonBondedForces->compute();		


		}

	}

	cudaEventRecord(stop);

	//**************** restart data ********************* //
	copy_lattice_d2h();
	map<string,double> info_ = map_list_of ("lx", l.x)("ly", l.y)("n", N)
                                         ("nblocks",nblocks)("nthreads", nthreads)
                                         ("num_of_strain_applied", num_of_strain_applied)
                                         ("rc", rc)("strain_rate", strain_rate)
                                         ("tstep", tsteps_max);

	write_restart_data(N,prefix,info_);

	float ms=0.0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("time loop took: %f ms\n",ms);


	return 0;
}
