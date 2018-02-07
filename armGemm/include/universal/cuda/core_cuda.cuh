#ifndef HPC_UNIVERSAL_CORE_CORE_CUH_
#define HPC_UNIVERSAL_CORE_CORE_CUH_

#include "warp_cuda.cuh"
#include "atomic_cuda.cuh"
#include "simd_functions_cuda.cuh"

template<typename T>
__host__ __device__ inline T divUp(T a, T b){
    return (a + b-1)/b;
}

#endif
