/**
 * @file scalar.hpp
 *
 * @brief scalar related operations
 *
 * @note Only support X86, ARM and CUDA, OCL is not supported
 */

#ifndef HPC_FMATH_SCALAR_H_
#define HPC_FMATH_SCALAR_H_

namespace HPC { namespace fmath {
#if defined(USE_X86)
    template<typename T>
        T x86ScalarSigmoid(T a);
#endif
#if defined(USE_ARM)
    template<typename T>
        T armScalarSigmoid(T a);
#endif
#if defined(USE_CUDA)
    template<typename T>
        T cudaScalarSigmoid(T a);
#endif

} };
#endif
