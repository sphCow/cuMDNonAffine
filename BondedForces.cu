#include "BondedForces.h"

//********************* KERNELS BEGIN ***********************************//

//**************** PBC KERNELS ******************************************//
__device__ __forceinline__ double d_pbcx(double xx) { return (xx - dev_l.x * rint(xx/dev_l.x) );}
__device__ __forceinline__ double d_pbcy(double yy) { return (yy - dev_l.y * rint(yy/dev_l.y) );}

template <int BLOCK_THREADS>
__global__
void kernel_non_bonded_force(const double4* __restrict__ dev_r,
                             const long* __restrict__ dev_pid,
                             const long* __restrict__ dev_nebz_pid,
                             const long* __restrict__ dev_pid2gtid,
                             const long N,
                             const long num_nebz,
                             const double k,
                             const double r0,
                             double4* dev_f,     // force [per particle]
                             double* dev_pe_reduced,        // pe, [reduced over blocks]
                             double4* dev_virial_tensor_pp,     // 4 component virial tensor [per particle]
                             double4* dev_virial_tensor_reduced, // 4 compoent virial tensor [reduced over blocks]
                             double* dev_virial_pressure_reduced     // sigma = sigma_xx + sigma_yy [reduced over blocks]
                             ) {

  long gtid = blockDim.x*blockIdx.x + threadIdx.x;
  if(gtid >= N) return;

  long my_pid, my_nebz_pid, my_nebz_gtid;
  double4 my_r, my_nebz_r;
  double4 my_f = make_double4(0.0,0.0,0.0,0.0);

  //my_thermo contribution
  double4 my_pe_virial_pressure = make_double4(0.0,0.0,0.0,0.0); //x -> pe, y -> virial_pressure
  double4 my_virial_tensor = make_double4(0.0, 0.0, 0.0, 0.0);

  my_r = dev_r[gtid];
  my_pid = dev_pid[gtid];

  for(int n=0; n<num_nebz; n++) {
    my_nebz_pid = dev_nebz_pid[num_nebz*my_pid+n];
    my_nebz_gtid = dev_pid2gtid[my_nebz_pid];
    my_nebz_r = dev_r[my_nebz_gtid];

    double dx = d_pbcx(my_r.x - my_nebz_r.x);
    double dy = d_pbcy(my_r.y - my_nebz_r.y);
    double dr = sqrt(dx*dx + dy*dy);

	// add strength factor here
  //  double fac = -k*(1.0 - r0/dr);
  //
 //   my_f.x -= fac*dx;
 //   my_f.y -= fac*dy;

    double fac = -k*(1.0 - r0/dr);

    my_f.x += fac*dx;
    my_f.y += fac*dy;
  
    my_pe_virial_pressure.x += 0.5*k*(dr-r0)*(dr-r0);

    my_virial_tensor.x += 0.5*fac*dx*dx; //xx
    my_virial_tensor.y += 0.5*fac*dx*dy; // xy
    my_virial_tensor.z += 0.5*fac*dy*dx; //yx
    my_virial_tensor.w += 0.5*fac*dy*dy; //yy

    my_pe_virial_pressure.y += my_virial_tensor.x + my_virial_tensor.w;
  }

  // ******************* WRITE PP DATA TO GLOBAL MEMORY ******************** //
  dev_f[gtid] += my_f;
  dev_virial_tensor_pp[gtid] += my_virial_tensor;

  //reduce over all threads in current block for average thermo quantities
  typedef cub::BlockReduce<double4, BLOCK_THREADS> BlockReduceDouble4;
  __shared__ typename BlockReduceDouble4::TempStorage temp_storage_double4;

  // ******************* PARTIAL REDUCTION OVER THREADS ******************** //
  double4 reduced; 
  // potential energy & virial pressure
  reduced = BlockReduceDouble4(temp_storage_double4).Reduce(my_pe_virial_pressure, Sum4());
  if(threadIdx.x == 0) dev_pe_reduced[blockIdx.x] += reduced.x;
  if(threadIdx.x == 0) dev_virial_pressure_reduced[blockIdx.x] += reduced.y;

  // mean virial tensor
  if(gtid<N) reduced = BlockReduceDouble4(temp_storage_double4).Reduce(my_virial_tensor, Sum4());
  if(threadIdx.x == 0) dev_virial_tensor_reduced[blockIdx.x] += reduced;




}

//************************ KERNELS END **********************************//



/************************ INTERPRET ****************************************/
int BondedForces::interpret(std::vector<std::string> tokenized) {
  std::cout << "|-------------------- Bonded Forces --------------------|" << std::endl;
  
  // BondedForces bond_type harmonic 1.0 0.4 neb_type triangular Nx Ny
  //      0          1      2   3    5

  std::vector<std::string>::iterator it;

  it = find(tokenized.begin(), tokenized.end(), "bond_type");

  if (it == tokenized.end()) {
    std::cerr << "BondedForces: you should define a bond_type." << std::endl;
  } else {
    int pos = std::distance(tokenized.begin(), it);
    if(tokenized[pos+1] == "harmonic") {
      bond_type = HARMONIC;
      k = boost::lexical_cast<double>(tokenized[pos+2]);
      r0 = boost::lexical_cast<double>(tokenized[pos+3]);
      std::cout << "BondedForces: Harmonic bonds with k = " << k << " r0 = " << r0 << std::endl;
    }

    //else if() {} // handle other bond types

    else {
      bond_type = BONDTYPE_UNDEFINED;
      std::cerr << "bond_type " << tokenized[pos+1] << " not known!" << std::endl;
    }
  }

  it = find(tokenized.begin(), tokenized.end(), "neb_type");

  if (it == tokenized.end()) {
    std::cerr << "BondedForces: you should define a neb_type." << std::endl;
  } else {
    int pos = std::distance(tokenized.begin(), it);
    if(tokenized[pos+1] == "triangular") {
        neb_type = TRIANGULAR;
        Nx = boost::lexical_cast<long>(tokenized[pos+2]);
        Ny = boost::lexical_cast<long>(tokenized[pos+3]);
        std::cout << "BondedForces: Triangular nebz with Nx = " << Nx << " Ny = " << Ny << std::endl;
    }

    // else if()

    else {
      neb_type = NEBTYPE_UNDEFINED;
      std::cerr << "neb_type " << tokenized[pos+1] << " not known!" << std::endl;
    }
  }

	

}

void BondedForces::init() {

  // number of neghibours
  switch(neb_type) {
    case TRIANGULAR:
      num_nebz = 6;

      //allocate
      nebz_pid = new long[num_nebz*N];
      cudaMalloc((void **) &dev_nebz_pid, N*num_nebz*sizeof(*dev_nebz_pid));
	  std::cout << "BondedForces: init N = " << N << std::endl;

      for(long ix=0; ix<Nx; ix++) {
        for(long iy=0; iy<Ny; iy++) {
          long i=ix+iy*Nx;
          if(iy%2 == 0) {
            nebz_pid[num_nebz*i+0] = ((ix + Nx - 1) % Nx + ((iy + Ny + 1) % Ny) * Nx); //NW
            nebz_pid[num_nebz*i+1] = ((ix + Nx - 1) % Nx + ((iy + Ny - 1) % Ny) * Nx); //SW
          }
          else {
            nebz_pid[num_nebz*i+0] = ((ix + 1) % Nx + ((iy + Ny + 1) % Ny) * Nx); //NE
            nebz_pid[num_nebz*i+1] = ((ix + 1) % Nx + ((iy + Ny - 1) % Ny) * Nx); //SE
          }

          nebz_pid[num_nebz*i+2] = ((ix + 1) % Nx + ((iy + Ny) % Ny) * Nx); //EAST
          nebz_pid[num_nebz*i+3] = ((ix + Nx) % Nx + ((iy + Ny + 1) % Ny) * Nx); //N
          nebz_pid[num_nebz*i+4] = ((ix + Nx - 1) % Nx + ((iy + Ny) % Ny) * Nx); //W
          nebz_pid[num_nebz*i+5] = ((ix + Nx) % Nx + ((iy + Ny - 1) % Ny) * Nx); //S
        }
      }

      //copy h2d
      cudaMemcpy(dev_nebz_pid, nebz_pid, N*num_nebz*sizeof(*dev_nebz_pid), cudaMemcpyHostToDevice);


    //handle other bond types
    //case SOME_OTHER_BOND_TYPE:
  }
}

void BondedForces::set_box_dim(const double &_lx, const double &_ly) {
  this->lx = _lx; this->ly = _ly;
  std::cout << "BondedForces: L set to " << lx << " " << ly << std::endl;

  double3 l = make_double3(lx, ly, 0.0);
  cudaMemcpyToSymbol(dev_l, &l, sizeof(l));
}

void BondedForces::compute() {
  map_pid2gtid(dev_pid,dev_pid2gtid,N,nblocks,nthreads);

  //TODO FIX THE TEMPLATE PARAMETER
  kernel_non_bonded_force<128><<<nblocks, nthreads>>>(dev_r,
                          dev_pid,
                          dev_nebz_pid,
                          dev_pid2gtid,
                          N,
                          num_nebz,
                          k,
                          r0,
                          dev_f,     // force [per particle]
                          dev_pe_reduced,        // pe, [reduced over blocks]
                          dev_virial_tensor_pp,     // 4 component virial tensor [per particle]
                          dev_virial_tensor_reduced, // 4 compoent virial tensor [reduced over blocks]
                          dev_virial_pressure_reduced);


}
