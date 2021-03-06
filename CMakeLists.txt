### CMakeLists.txt for CUDA

cmake_minimum_required(VERSION 2.6)
project(cuMD)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -arch=sm_20
    )

# Specify include directories
include_directories(
    /home/pnath/opt/include
    /home/pnath/softwares/cub-1.3.2
    )

# Specify library paths
link_directories(

    )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
	cuMD.x

  FileIO.h
	global.cpp
	global.h
	cuda_constants.cuh
	precision.cuh
	mdcuda.cu
	FileIO.cpp
	nonaffine.cu
	nonaffine.h
	thermostat.h
	thermostat.cu
	ThermostatBDP.cpp
	ThermostatBDP.h
  kernels/map_pid2gtid.cu
  kernels/reduction.cu
  BondedForces.h
  BondedForces.cu
  NonBondedForces.cu
  NonBondedForces.h

    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
libcudart    )
