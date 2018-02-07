#if defined(USE_CUDA)
#ifndef HPC_UNIVERSAL_CORE_ATOMIC_CUH_
#define HPC_UNIVERSAL_CORE_ATOMIC_CUH_

//#if (GPU_ARCH < 200)
//#error GPU capacity must larger than 2.0
//#endif

/**
 * @brief atomicAdd
 * support int/uint/long/float/double
 *
 **/

template<typename T> __device__ __forceinline__ T atomicAdd(T* address, T val) {
    return ::atomicAdd(address, val);
}

template<> __device__ double __forceinline__ atomicAdd<double>(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

template<typename T> __device__ T __forceinline__ atomicMin(T* address, T val) {
    return ::atomicMin(address, val);
}

template<> __device__ float __forceinline__ atomicMin<float>(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

template<> __device__ double __forceinline__ atomicMin<double>(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_ull, assumed,
                __double_as_longlong(::fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

template<typename T> __device__ __forceinline__ T atomicMax(T* address, T val) {
    return ::atomicMax(address, val);
}

template<> __device__ __forceinline__ float atomicMax(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


template<> __device__ __forceinline__ double atomicMax<double>(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_ull, assumed,
                __double_as_longlong(::fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#endif
#endif
