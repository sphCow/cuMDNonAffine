#ifndef __MAP_PID2GTID_CU__
#define __MAP_PID2GTID_CU__

__global__
void kernel_map_pid2gtid(long *dev_pid, long *dev_pid2gtid, const long N)  {
  long gtid = blockDim.x*blockIdx.x + threadIdx.x;
	if(gtid >= N) return;

  long my_pid = dev_pid[gtid];
  dev_pid2gtid[my_pid] = gtid;

}

extern "C" void map_pid2gtid(long *dev_pid,
                       long *dev_pid2gtid,
                       const long &N ,
                       const int &nblocks,
                       const int &nthreads)  {

  kernel_map_pid2gtid<<<nblocks,nthreads>>>(dev_pid, dev_pid2gtid, N);


}

#endif
