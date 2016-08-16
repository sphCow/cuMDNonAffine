#ifndef PRECISION
#define PRECISION

#define DOUBLE_PRECISION

	#ifdef SINGLE_PRECISION
		#define real float
		#define real4 float4
		#define real3 float3
		#define make_real3 make_float3
		#define make_real4 make_float4
	#endif

	#ifdef DOUBLE_PRECISION
		#define real double
		#define real4 double4
		#define real3 double3
		#define make_real3 make_double3
		#define make_real4 make_double4
		//# define %f %lf
	#endif

	#define CUDA_CHECK_RETURN(value) {											\
		cudaError_t _m_cudaStat = value;										\
		if (_m_cudaStat != cudaSuccess) {										\
			fprintf(stderr, "Error %s at line %d in file %s\n",					\
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
			exit(1);															\
		} }

	#define CUDA_CHECK CUDA_CHECK_RETURN(cudaThreadSynchronize());CUDA_CHECK_RETURN(cudaGetLastError());


#endif
