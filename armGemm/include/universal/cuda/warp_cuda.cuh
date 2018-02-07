#if defined(USE_CUDA)

#ifndef HPC_UNIVERSAL_CORE_WARP_CUH_
#define HPC_UNIVERSAL_CORE_WARP_CUH_

/**
 * @brief scan in warp per numThreads
 **/
template<typename T> __device__ T inclusiveScanInWarp(T v, int id, int numThreads){
    for(int i = 1; i < numThreads; i *= 2){
        T d = __shfl_up(v, i);
        if(id >= i) v += d;
    }
    return v;
}

/**
 * @brief scan in warp
 **/
template<typename T> __device__ T inclusiveWarpScan(T v, int idInWarp){
    return inclusiveScanInWarp(v, idInWarp, warpSize);
}

#endif
#endif
