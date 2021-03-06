#include "NonBondedForces.h"

// *************************** KERNELS *************************************// 
__global__ 
void populate_cell_nebz(uint2 *c, uint4 *cn1, uint4 *cn2) {
	uint gtid = blockDim.x*blockIdx.x + threadIdx.x;

	if(gtid < dev_ncell.z) {
		uint iy = uint(gtid/dev_ncell.x);
		uint ix = gtid % dev_ncell.x;

    	cn1[gtid].x = ((ix + 1) % dev_ncell.x + ((iy + dev_ncell.y) % dev_ncell.y) * dev_ncell.x); //EAST
    	cn1[gtid].y = ((ix + 1) % dev_ncell.x + ((iy + dev_ncell.y + 1) % dev_ncell.y) * dev_ncell.x); //NE
    	cn1[gtid].z = ((ix + dev_ncell.x) % dev_ncell.x + ((iy + dev_ncell.y + 1) % dev_ncell.y) * dev_ncell.x); //N
    	cn1[gtid].w = ((ix + dev_ncell.x - 1) % dev_ncell.x + ((iy + dev_ncell.y + 1) % dev_ncell.y) * dev_ncell.x); //NW

		cn2[gtid].x = ((ix + dev_ncell.x - 1) % dev_ncell.x + ((iy + dev_ncell.y) % dev_ncell.y) * dev_ncell.x); //W
		cn2[gtid].y = ((ix + dev_ncell.x - 1) % dev_ncell.x + ((iy + dev_ncell.y - 1) % dev_ncell.y) * dev_ncell.x); //SW
		cn2[gtid].z = ((ix + dev_ncell.x) % dev_ncell.x + ((iy + dev_ncell.y - 1) % dev_ncell.y) * dev_ncell.x); //S
		cn2[gtid].w = ((ix + 1) % dev_ncell.x + ((iy + dev_ncell.y - 1) % dev_ncell.y) * dev_ncell.x); //SE
	}
}

/********************************************************************************
// [GPU] Build/update cell_list								   				  //
// fill-in dev_particles->cid array with cell ids 	           				  //
/********************************************************************************/
__global__ 
void get_cid(const double4 __restrict__ *r, long *cid, int *clen) {
	long gtid = blockDim.x*blockIdx.x + threadIdx.x;

	if(gtid >= dev_N) return;

	long my_cx = long((r[gtid].x + 0.5*dev_l.x) / dev_lcell.x);
	long my_cy = long((r[gtid].y + 0.5*dev_l.y) / dev_lcell.y);
  	long my_cid = (my_cx) + (my_cy) * (dev_ncell.x);

    cid[gtid] = my_cid;

    // ?? CAN WE IMPROVE THIS ATOMIC ADD HERE ?? //
    atomicAdd(&(clen[my_cid]),(int)1);
    // ?? IDEA ?? //

    if(my_cid > dev_ncell.z-1) printf("Warn!! i'm out of the box!!");
}

/****************************************************************************/
// [GPU] Compute LJ force on i-th particle due to j-th particle			//
//
/****************************************************************************/
__device__ 
void get_pair_lj(const double4 __restrict__ *r_i,
				 const double4 __restrict__ *r_j,
				 double4 __restrict__ *f_i,
				 double4 __restrict__ *pe,
				 double4 __restrict__ *virial_ij,
				 double3 __restrict__ *born_ij) {

	double xx,yy,rr,rri,r6i,ff,r14i,r8i;
	//double x2,y2;

	xx = r_i->x - r_j->x;
	yy = r_i->y - r_j->y;

	//pbc
	xx -= dev_l.x * rint(xx/dev_l.x);
	yy -= dev_l.y * rint(yy/dev_l.y);

	// born - phi
	//double phi;

	//squared distance between i & j-th particle
	rr = xx*xx + yy*yy;

	//printf("l %d%f %f %f %f %f %f",dev_N,dev_l.x,dev_l.y,dev_rc,dev_rc2,dev_lcell.x,dev_lcell.y);
	if(rr < dev_rc2) {
		rri = 1.0/rr;
		r6i = rri*rri*rri;
    	ff = 48.0*dev_strength*rri*r6i*(r6i-0.5); // -dU/dr * 1/r

    	f_i->x += ff*xx;
    	f_i->y += ff*yy;

    	pe->x += dev_strength*0.5*4.0*r6i*(r6i-1.0);
		pe->y += 0.5*ff*rr;

		virial_ij->x += 0.5*ff * xx * xx; //xx
		virial_ij->y += 0.5*ff * yy * xx; //xy
    	virial_ij->z += 0.5*ff * xx * yy; //yx
    	virial_ij->w += 0.5*ff * yy * yy; //yy

		// REQUIRED FOR BORN
		//r8i = r6i*rri;
		//r14i = r8i*r6i;
		//phi = rri * (672.0 * r14i - 192.0 * r8i);

		//born_ij->x += 0.5*xx * xx * xx * xx * phi; // 1111
    	//born_ij->y += 0.5*xx * xx * yy * yy * phi; // 1122
    	//born_ij->z += 0.5*yy * yy * yy * yy * phi;

	}
}

template <int BLOCK_THREADS>
__global__ 
void calculate_force_with_cell_list(const double4 __restrict__ *r_,
									double4 *f_,
									const long  *pid_,
									const long   *cid_,
									const int   *clen,
									const int   *cbegin,
									const uint4 __restrict__ *cl_neb1,
									const uint4 __restrict__ *cl_neb2,
									double* __restrict__ dev_pe_reduced,        // pe, [reduced over blocks]
                             		double4* __restrict__ dev_virial_tensor_pp,     // 4 component virial tensor [per particle]
                             		double4* __restrict__ dev_virial_tensor_reduced, // 4 compoent virial tensor [reduced over blocks]
                             		double* __restrict__ dev_virial_pressure_reduced     // sigma = sigma_xx + sigma_yy [reduced over blocks]
									) {	

	long gtid = blockDim.x*blockIdx.x + threadIdx.x;
	if(gtid >= dev_N) return;

	//if(gtid==1000) printf("l %d %f %f %f %f %f %f\n",dev_N,dev_l.x,dev_l.y,dev_rc,dev_rc2,dev_lcell.x,dev_lcell.y);

	// shared memory for thermo reductions
	//extern __shared__ double shared_thermo[];

	//define registers first
	double4 r_i, r_j;
	double4 f_i = make_double4(0.0,0.0,0.0,0.0);
	long cid_i;
	ulong pid_i, pid_j;
	uint4 neb1, neb2;
	long i;

	//thermo
	double4 my_pe_virial_pressure = make_double4(0.0,0.0,0.0,0.0);
	double4 my_virial_tensor = make_double4(0.0,0.0,0.0,0.0);
	double3 born_ij = make_double3(0.0,0.0,0.0);

	//read from global memory
	r_i = r_[gtid];
	pid_i = pid_[gtid];
	cid_i = cid_[gtid];

	//populate neb ids
	neb1 = cl_neb1[cid_i];
	neb2 = cl_neb2[cid_i];

	/*
	//loop over 8 nebs
	for(i=cbegin[neb1.x]; i<cbegin[neb1.x]+clen[neb1.x]; i++) get_pair_lj_(&r_i, &r_[i],&f_i, &my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	for(i=cbegin[neb1.y]; i<cbegin[neb1.y]+clen[neb1.y]; i++) get_pair_lj_(&r_i, &r_[i],&f_i, &my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	for(i=cbegin[neb1.z]; i<cbegin[neb1.z]+clen[neb1.z]; i++) get_pair_lj_(&r_i, &r_[i],&f_i, &my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	for(i=cbegin[neb1.w]; i<cbegin[neb1.w]+clen[neb1.w]; i++) get_pair_lj_(&r_i, &r_[i],&f_i, &my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	for(i=cbegin[neb2.x]; i<cbegin[neb2.x]+clen[neb2.x]; i++) get_pair_lj_(&r_i, &r_[i],&f_i, &my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	for(i=cbegin[neb2.y]; i<cbegin[neb2.y]+clen[neb2.y]; i++) get_pair_lj_(&r_i, &r_[i],&f_i, &my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	for(i=cbegin[neb2.z]; i<cbegin[neb2.z]+clen[neb2.z]; i++) get_pair_lj_(&r_i, &r_[i],&f_i, &my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	for(i=cbegin[neb2.w]; i<cbegin[neb2.w]+clen[neb2.w]; i++) get_pair_lj_(&r_i, &r_[i],&f_i, &my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	
	//loop over particles in its own cell
	for(i=cbegin[cid_i]; i<cbegin[cid_i]+clen[cid_i]; i++) {
		r_j = r_[i];
		pid_j = pid_[i];
		if(pid_i != pid_j) {
			get_pair_lj_(&r_i, &r_j,&f_i,&my_pe_virial_pressure, &my_virial_tensor, &born_ij);
		}
	}
	*/
	
	//loop over 8 nebs
	for(i=cbegin[neb1.x]; i<cbegin[neb1.x]+clen[neb1.x]; i++) {
		r_j = r_[i];
		get_pair_lj(&r_i, &r_j,&f_i,&my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	}

	for(i=cbegin[neb1.y]; i<cbegin[neb1.y]+clen[neb1.y]; i++) {
		r_j = r_[i];
		get_pair_lj(&r_i, &r_j ,&f_i ,&my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	}

	for(i=cbegin[neb1.z]; i<cbegin[neb1.z]+clen[neb1.z]; i++) {
		r_j = r_[i];
		get_pair_lj(&r_i, &r_j,&f_i,&my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	}

	for(i=cbegin[neb1.w]; i<cbegin[neb1.w]+clen[neb1.w]; i++) {
		r_j = r_[i];
		get_pair_lj(&r_i, &r_j,&f_i,&my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	}

	for(i=cbegin[neb2.x]; i<cbegin[neb2.x]+clen[neb2.x]; i++) {
		r_j = r_[i];
		get_pair_lj(&r_i, &r_j,&f_i,&my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	}

	for(i=cbegin[neb2.y]; i<cbegin[neb2.y]+clen[neb2.y]; i++) {
		r_j = r_[i];
		get_pair_lj(&r_i, &r_j,&f_i,&my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	}

	for(i=cbegin[neb2.z]; i<cbegin[neb2.z]+clen[neb2.z]; i++) {
		r_j = r_[i];
		get_pair_lj(&r_i, &r_j,&f_i,&my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	}

	for(i=cbegin[neb2.w]; i<cbegin[neb2.w]+clen[neb2.w]; i++) {
		r_j = r_[i];
		get_pair_lj(&r_i, &r_j,&f_i,&my_pe_virial_pressure, &my_virial_tensor, &born_ij);
	}

	//loop over particles in its own cell
	for(i=cbegin[cid_i]; i<cbegin[cid_i]+clen[cid_i]; i++) {
		r_j = r_[i];
		pid_j = pid_[i];
		if(pid_i != pid_j) {
			get_pair_lj(&r_i, &r_j,&f_i,&my_pe_virial_pressure, &my_virial_tensor, &born_ij);
		}
	}

	// ******************* WRITE PP DATA TO GLOBAL MEMORY ******************** //
	f_[gtid] += f_i; //todo
  	dev_virial_tensor_pp[gtid] += my_virial_tensor; //todo

	//reduce over all threads in current block for average thermo quantities
  	typedef cub::BlockReduce<double4, BLOCK_THREADS, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduceDouble4;
  	__shared__ typename BlockReduceDouble4::TempStorage temp_storage_double4;
	
	// ******************* PARTIAL REDUCTION OVER THREADS ******************** //
	double4 reduced;
	// potential energy & virial pressure
  	reduced = BlockReduceDouble4(temp_storage_double4).Reduce(my_pe_virial_pressure, Sum4());
  	if(threadIdx.x == 0) dev_pe_reduced[blockIdx.x] += reduced.x;
  	if(threadIdx.x == 0) dev_virial_pressure_reduced[blockIdx.x] += reduced.y; 
  	
  	// mean virial tensor
  	reduced = BlockReduceDouble4(temp_storage_double4).Reduce(my_virial_tensor, Sum4());
  	if(threadIdx.x == 0) dev_virial_tensor_reduced[blockIdx.x] += reduced; 
}
/* ****************************************************************************
   ********************************** KERNELS END *****************************
   ****************************************************************************/

int NonBondedForces::interpret(std::vector<std::string> tokenized) {
  // NonBondedForces interaction_type lj 1.0 1.0 cut_off 2.5 [strength 0.8] 
  //      0                  1        2   3   5
  std::cout << std::endl << "|------------- Non-Bonded Forces -------------|" << std::endl;	

  std::vector<std::string>::iterator it;

  it = find(tokenized.begin(), tokenized.end(), "interaction_type");

  if (it == tokenized.end()) {
    std::cerr << "NonBondedForces: you should define an interaction_type." << std::endl;
  } else {
    int pos = std::distance(tokenized.begin(), it);
    if(tokenized[pos+1] == "lj") {
      interaction_type = LJ;
      sigma = boost::lexical_cast<double>(tokenized[pos+2]); 
      epsilon = boost::lexical_cast<double>(tokenized[pos+3]);
      std::cout << "NonBondedForces: Lennard-Jones interaction with sigma = " << sigma << " epsilon = " << epsilon << std::endl;
    }

    //else if() {} // handle other bond types

    else {
      interaction_type = INTERACTION_TYPE_UNDEFINED;
      std::cerr << "interaction_type " << tokenized[pos+1] << " not known!" << std::endl;
    }
  }

  it = find(tokenized.begin(), tokenized.end(), "cut_off");

  if (it == tokenized.end()) {
    std::cerr << "NonBondedForces: you should specify a cut-off" << std::endl;
  } else {
    int pos = std::distance(tokenized.begin(), it);
    rc = boost::lexical_cast<double>(tokenized[pos+1]);
    std::cout << "NonBondedForces: Cut-off set to rc = " << rc << std::endl;
    }

    // else if()
  
  // STRENGTH  
  it = find(tokenized.begin(), tokenized.end(), "strength");
  
  if (it == tokenized.end()) {
    std::cout << "NonBondedForces: No strength specified. using default." << std::endl;
    strength = 1.0;
  } else {
    int pos = std::distance(tokenized.begin(), it);
    strength = boost::lexical_cast<double>(tokenized[pos+1]);
    
    	if(strength < 0.0 || strength > 1.0) {
    		std::cout << "NonBondedForces: invalid strength specified. Using default" << std::endl;	
    		strength = 1.0;
    	}
    }
    
    cudaMemcpyToSymbol(dev_strength, &strength, sizeof(double));
	std::cout << "NonBondedForces: strength = " << strength << std::endl;
	std::cout << "|---------------------------------------------|" << std::endl;	

}

void NonBondedForces::host_allocate_celllist() {
	host_cell_list = new uint2[ncell.z]();
	host_cell_nebz1 = new uint4[ncell.z]();
	host_cell_nebz2 = new uint4[ncell.z]();
	host_clen = new int[ncell.z]();
	host_cbegin = new int[ncell.z]();
}

void NonBondedForces::dev_allocate_celllist() {
	cudaMalloc((void**) &dev_cell_list, ncell.z*sizeof(uint2));
	cudaMalloc((void**) &dev_cell_nebz1, ncell.z*sizeof(uint4));
	cudaMalloc((void**) &dev_cell_nebz2, ncell.z*sizeof(uint4));
	cudaMalloc((void**) &dev_clen, ncell.z*sizeof(int));
	cudaMalloc((void**) &dev_cbegin, ncell.z*sizeof(int));
}

void NonBondedForces::host_set_cell_params() {
	ncell.x = (uint)floor(l.x / (rc));  //2;
  	ncell.y = (uint)floor(l.y / (rc));  //2;

  	ncell.z = ncell.x * ncell.y;

  	lcell.x = l.x / double(ncell.x);
  	lcell.y = l.y / double(ncell.y);

  	//std::cout << "NonBondedForces parameters for cell list is set. " << N <<  std::endl;
 	//std::cout << "NonBondedForces rc -> " << rc << std::endl;
  	//std::cout << "NonBondedForces l -> " << l.x << "x" << l.y << std::endl;
  	//std::cout << "NonBondedForces ncell -> " << ncell.x << "x" << ncell.y << " = " << ncell.z << std::endl;
  	//std::cout << "NonBondedForces lcell -> " << lcell.x << "x" << lcell.y << std::endl;
}

void NonBondedForces::copy_celllist_d2h() {
	cudaMemcpy(host_cell_list, dev_cell_list, ncell.z*sizeof(uint2), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_cell_nebz1, dev_cell_nebz1, ncell.z*sizeof(uint4), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_cell_nebz2, dev_cell_nebz2, ncell.z*sizeof(uint4), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_clen, dev_clen, ncell.z*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_cbegin, dev_cbegin, ncell.z*sizeof(int), cudaMemcpyDeviceToHost);
}

/*
void NonBondedForces::print_celllist() {
	fprintf(stderr, "cell_id \t start end \t nebz \n");
	for(uint i=0; i<ncell.z; i++) {
		fdcl << 
			i << " " << host_cbegin[i] << " " << host_clen[i] << " " <<
			host_cell_nebz1[i].x<< " " <<host_cell_nebz1[i].y<< " " <<host_cell_nebz1[i].z<< " " <<host_cell_nebz1[i].w<< " " <<
			host_cell_nebz2[i].x<< " " <<host_cell_nebz2[i].y<< " " <<host_cell_nebz2[i].z<< " " <<host_cell_nebz2[i].w << std::endl;
	}
	fdcl << std::endl << std::endl;
}

//__constant__ long dev_N;
//__constant__ double dev_rc;
//__constant__ double dev_rc2;dev_l, dev_lcell, dev_ncell

*/
/************************************************************************///
// [GPU] [CPU] Sort 								   //
// http://stackoverflow.com/questions/23541503/sorting-arrays-of-structures-in-cuda //
/****************************************************************************/
void NonBondedForces::sort_by_cid() {

	thrust::device_ptr<long> dev_ptr_pid = thrust::device_pointer_cast(dev_pid);
	thrust::device_ptr<long> dev_ptr_cid = thrust::device_pointer_cast(dev_cid);
	thrust::device_ptr<double4> dev_ptr_r = thrust::device_pointer_cast(dev_r);
	thrust::device_ptr<double4> dev_ptr_v = thrust::device_pointer_cast(dev_v);
	thrust::device_ptr<double4> dev_ptr_f = thrust::device_pointer_cast(dev_f);


	//for cell list
	thrust::device_ptr<int> dev_ptr_clen = thrust::device_pointer_cast(dev_clen);
	thrust::device_ptr<int> dev_ptr_cbegin = thrust::device_pointer_cast(dev_cbegin);

	thrust::sort_by_key(dev_ptr_cid, dev_ptr_cid + N,
		thrust::make_zip_iterator(
			thrust::make_tuple( dev_ptr_pid,dev_ptr_r, dev_ptr_v, dev_ptr_f)));

	// exclusive_scan to get start ids
	thrust::exclusive_scan(dev_ptr_clen, dev_ptr_clen+ncell.z, dev_ptr_cbegin);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());

	//get_cell_start_ids<<<nblocks, nthreads>>>(dev_cid, dev_cbegin, dev_clen);
	//get_cell_len<<<nblocks, nthreads>>>(dev_cid, dev_cbegin, dev_clen);

}



void NonBondedForces::set_box_dim(const double &lx, const double &ly) {

	l.x = lx;
	l.y = ly;
	rc2 = rc*rc;
	
	cudaMemcpyToSymbol(dev_l, &l, sizeof(l));
	cudaMemcpyToSymbol(dev_N, &N, sizeof(N));
	cudaMemcpyToSymbol(dev_rc, &rc, sizeof(rc));
	cudaMemcpyToSymbol(dev_rc2, &rc2, sizeof(rc2));
	
	// set cell parameters in host
	host_set_cell_params(); // sets ncell, lcell

	cudaMemcpyToSymbol(dev_lcell, &lcell, sizeof(lcell));
	cudaMemcpyToSymbol(dev_ncell, &ncell, sizeof(ncell));

	// allocate mem for celllist in dev & host
	dev_allocate_celllist(); //allocates dev_cell_list, dev_cell_nebz1 &2
	//host_allocate_celllist(); //DEBUG
	
	populate_cell_nebz<<<nblocks,nthreads>>>(dev_cell_list,dev_cell_nebz1,dev_cell_nebz2);
	CUDA_CHECK_RETURN(cudaThreadSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	
	//cudaMemset(dev_clen, 0, ncell.z*sizeof(int));
	
	//get_cid<<<nblocks,nthreads>>>(dev_r, dev_cid, dev_clen);//
	//CUDA_CHECK_RETURN(cudaThreadSynchronize());
	//CUDA_CHECK_RETURN(cudaGetLastError());
	
	//sort_by_cid();

}

void NonBondedForces::compute() {
	
	cudaMemset(dev_clen, 0, ncell.z*sizeof(int));
	
	get_cid<<<nblocks,nthreads>>>(dev_r, dev_cid, dev_clen);//
	CUDA_CHECK_RETURN(cudaThreadSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	
	sort_by_cid();
	//map_pid2gtid(dev_pid,dev_pid2gtid,N,nblocks,nthreads);
	
	calculate_force_with_cell_list<128><<<nblocks, nthreads>>>(dev_r,
									dev_f,
									dev_pid,
									dev_cid,
									dev_clen,
									dev_cbegin,
									dev_cell_nebz1,
									dev_cell_nebz2,
									dev_pe_reduced,        // pe, [reduced over blocks]
                             		dev_virial_tensor_pp,     // 4 component virial tensor [per particle]
                             		dev_virial_tensor_reduced, // 4 compoent virial tensor [reduced over blocks]
                             		dev_virial_pressure_reduced     // sigma = sigma_xx + sigma_yy [reduced over blocks]
									);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());
	CUDA_CHECK_RETURN(cudaGetLastError());
	
	//double non_bonded_pe = GlobalReduce(dev_pe_reduced, nblocks);
	//std::cout << non_bonded_pe << std::endl;
}


































