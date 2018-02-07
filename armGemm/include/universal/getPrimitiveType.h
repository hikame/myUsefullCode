#ifndef HPC_UNIVERSAL_GETPRIMITIVETYPE_H_
#define HPC_UNIVERSAL_GETPRIMITIVETYPE_H_

#include "primtypes.h"
#include "types.h"

#if defined(USE_CUDA) || defined(USE_OCL)
#include "half.h"
using namespace uni::half;
#endif

namespace HPC {
    namespace uni {
        template<typename T> inline PrimitiveType uniPrimitiveType();

        template<> inline PrimitiveType uniPrimitiveType<int8>(){return HPC_INT8;}
        template<> inline PrimitiveType uniPrimitiveType<int8x2>(){ return HPC_INT8x2;}
        template<> inline PrimitiveType uniPrimitiveType<int8x3>(){ return HPC_INT8x3;}
        template<> inline PrimitiveType uniPrimitiveType<int8x4>(){ return HPC_INT8x4;}

        template<> inline PrimitiveType uniPrimitiveType<uint8>(){ return HPC_UINT8;}
        template<> inline PrimitiveType uniPrimitiveType<uint8x2>(){ return HPC_UINT8x2;}
        template<> inline PrimitiveType uniPrimitiveType<uint8x3>(){ return HPC_UINT8x3;}
        template<> inline PrimitiveType uniPrimitiveType<uint8x4>(){ return HPC_UINT8x4;}

        template<> inline PrimitiveType uniPrimitiveType<int16>(){return HPC_INT16;}
        template<> inline PrimitiveType uniPrimitiveType<int16x2>(){ return HPC_INT16x2;}
        template<> inline PrimitiveType uniPrimitiveType<int16x3>(){ return HPC_INT16x3;}
        template<> inline PrimitiveType uniPrimitiveType<int16x4>(){ return HPC_INT16x4;}

        template<> inline PrimitiveType uniPrimitiveType<uint16>(){ return HPC_UINT16;}
        template<> inline PrimitiveType uniPrimitiveType<uint16x2>(){ return HPC_UINT16x2;}
        template<> inline PrimitiveType uniPrimitiveType<uint16x3>(){ return HPC_UINT16x3;}
        template<> inline PrimitiveType uniPrimitiveType<uint16x4>(){ return HPC_UINT16x4;}

        template<> inline PrimitiveType uniPrimitiveType<int32>(){return HPC_INT32;}
        template<> inline PrimitiveType uniPrimitiveType<int32x2>(){ return HPC_INT32x2;}
        template<> inline PrimitiveType uniPrimitiveType<int32x3>(){ return HPC_INT32x3;}
        template<> inline PrimitiveType uniPrimitiveType<int32x4>(){ return HPC_INT32x4;}

        template<> inline PrimitiveType uniPrimitiveType<uint32>(){ return HPC_UINT32;}
        template<> inline PrimitiveType uniPrimitiveType<uint32x2>(){ return HPC_UINT32x2;}
        template<> inline PrimitiveType uniPrimitiveType<uint32x3>(){ return HPC_UINT32x3;}
        template<> inline PrimitiveType uniPrimitiveType<uint32x4>(){ return HPC_UINT32x4;}

        template<> inline PrimitiveType uniPrimitiveType<float32>(){return HPC_FLOAT32;}
        template<> inline PrimitiveType uniPrimitiveType<float32x2>(){ return HPC_FLOAT32x2;}
        template<> inline PrimitiveType uniPrimitiveType<float32x3>(){ return HPC_FLOAT32x3;}
        template<> inline PrimitiveType uniPrimitiveType<float32x4>(){ return HPC_FLOAT32x4;}

        template<> inline PrimitiveType uniPrimitiveType<float64>(){return HPC_FLOAT64;}
        template<> inline PrimitiveType uniPrimitiveType<float64x2>(){ return HPC_FLOAT64x2;}
        template<> inline PrimitiveType uniPrimitiveType<float64x3>(){ return HPC_FLOAT64x3;}
        template<> inline PrimitiveType uniPrimitiveType<float64x4>(){ return HPC_FLOAT64x4;}

#if defined(USE_CUDA) || defined(USE_OCL)
        template<> inline PrimitiveType uniPrimitiveType<half_t>(){return HPC_FLOAT16;}
#endif
    }
}
#endif
