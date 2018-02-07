#ifndef HPC_UNIVERSAL_PRIMTYPES_H
#define HPC_UNIVERSAL_PRIMTYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>

/**
 * enumerator to indicate primitive numeric types
 */
typedef enum {
    HPC_BYTE     = 0,     ///< obscure bytes (e.g. encoded JPEG)
    HPC_BOOL     = 1,     ///< boolean (1-byte)

    HPC_INT8     = 11,     ///< 8-bit signed integer
    HPC_INT8x2   = 12,     ///< 8-bit signed integer
    HPC_INT8x3   = 13,     ///< 8-bit signed integer
    HPC_INT8x4   = 14,     ///< 8-bit signed integer

    HPC_INT16    = 21,     ///< 16-bit signed integer
    HPC_INT16x2  = 22,     ///< 16-bit signed integer
    HPC_INT16x3  = 23,     ///< 16-bit signed integer
    HPC_INT16x4  = 24,     ///< 16-bit signed integer

    HPC_INT32    = 31,     ///< 32-bit signed integer
    HPC_INT32x2  = 32,     ///< 32-bit signed integer
    HPC_INT32x3  = 33,     ///< 32-bit signed integer
    HPC_INT32x4  = 34,     ///< 32-bit signed integer

    HPC_INT64    = 4,     ///< 64-bit signed integer

    HPC_UINT8    = 51,     ///< 8-bit unsigned integer
    HPC_UINT8x2  = 52,     ///< 8-bit unsigned integer
    HPC_UINT8x3  = 53,     ///< 8-bit unsigned integer
    HPC_UINT8x4  = 54,     ///< 8-bit unsigned integer

    HPC_UINT16   = 61,     ///< 16-bit unsigned integer
    HPC_UINT16x2 = 62,     ///< 16-bit unsigned integer
    HPC_UINT16x3 = 63,     ///< 16-bit unsigned integer
    HPC_UINT16x4 = 64,     ///< 16-bit unsigned integer

    HPC_UINT32   = 71,     ///< 32-bit unsigned integer
    HPC_UINT32x2 = 72,     ///< 32-bit unsigned integer
    HPC_UINT32x3 = 73,     ///< 32-bit unsigned integer
    HPC_UINT32x4 = 74,     ///< 32-bit unsigned integer

    HPC_UINT64   = 8,     ///< 64-bit unsigned integer

    HPC_FLOAT16   = 91,   ///< 16-bit floating point real number
    HPC_FLOAT16x2 = 92,   ///< 16-bit floating point real number
    HPC_FLOAT16x3 = 93,   ///< 16-bit floating point real number
    HPC_FLOAT16x4 = 94,   ///< 16-bit floating point real number

    HPC_FLOAT32   = 101,   ///< 32-bit floating point real number
    HPC_FLOAT32x2 = 102,   ///< 32-bit floating point real number
    HPC_FLOAT32x3 = 103,   ///< 32-bit floating point real number
    HPC_FLOAT32x4 = 104,   ///< 32-bit floating point real number

    HPC_FLOAT64   = 111,   ///< 64-bit floating point real number
    HPC_FLOAT64x2 = 112,   ///< 64-bit floating point real number
    HPC_FLOAT64x3 = 113,   ///< 64-bit floating point real number
    HPC_FLOAT64x4 = 114,   ///< 64-bit floating point real number

    HPC_COMPLEX32   = 121,     ///< 32-bit complex number
    HPC_COMPLEX32x2 = 122,     ///< 32-bit complex number
    HPC_COMPLEX32x3 = 123,     ///< 32-bit complex number
    HPC_COMPLEX32x4 = 124,     ///< 32-bit complex number

    HPC_COMPLEX64   = 141,     ///< 64-bit complex number
    HPC_COMPLEX128  = 151     ///< 128-bit complex number
} PrimitiveType;

// convenient functions for working with primitive type enumerators

inline size_t uniPrimitiveTypeSize(PrimitiveType t) {
    switch (t) {
        case HPC_BYTE:
            return 1;
        case HPC_BOOL:
            return 1;
        case HPC_INT8:
            return 1;
        case HPC_INT16:
            return 2;
        case HPC_INT32:
            return 4;
        case HPC_INT64:
            return 8;
        case HPC_UINT8:
            return 1;
        case HPC_UINT16:
            return 2;
        case HPC_UINT32:
            return 4;
        case HPC_UINT64:
            return 8;
        case HPC_FLOAT16:
            return 2;
        case HPC_FLOAT32:
            return 4;
        case HPC_FLOAT64:
            return 8;
        case HPC_COMPLEX32:
            return 4;
        case HPC_COMPLEX64:
            return 8;
        case HPC_COMPLEX128:
            return 16;
        default:
            return 0;
    }
}

/**
 * @brief Get the size of the input primitive type
 * @note will deprecated in the near future
**/
inline size_t hpcPrimitiveTypeSize(PrimitiveType t) {
    return uniPrimitiveTypeSize(t);
}

inline const char* uniPrimitiveTypeName(PrimitiveType t) {
    switch (t) {
        case HPC_BYTE:
            return "byte";
        case HPC_BOOL:
            return "bool";
        case HPC_INT8:
            return "int8";
        case HPC_INT16:
            return "int16";
        case HPC_INT32:
            return "int32";
        case HPC_INT64:
            return "int64";
        case HPC_UINT8:
            return "uint8";
        case HPC_UINT16:
            return "uint16";
        case HPC_UINT32:
            return "uint32";
        case HPC_UINT64:
            return "uint64";
        case HPC_FLOAT16:
            return "float16";
        case HPC_FLOAT32:
            return "float32";
        case HPC_FLOAT64:
            return "float64";
        case HPC_COMPLEX32:
            return "complex32";
        case HPC_COMPLEX64:
            return "complex64";
        case HPC_COMPLEX128:
            return "complex128";
        default:
            return "unknown";
    }
}

/** Get the name of a primitive type
 * @note deprecated
 **/
inline const char* hpcPrimitiveTypeName(PrimitiveType t) {
    return uniPrimitiveTypeName(t);
}

#ifdef __cplusplus
}
#endif

#endif
