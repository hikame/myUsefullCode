#ifndef HPC_UNIVERSAL_DATASPEC_H_
#define HPC_UNIVERSAL_DATASPEC_H_

#include "primtypes.h"
#include "status.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

enum UniDataKind {
    UNI_OPAQUE = 0,   ///< Opaque byte-sequence (e.g. encoded JPEG)
    UNI_SCALAR = 1,   ///< A single scalar number
    UNI_ARRAY  = 2,    ///< A (dense) array (up to 4D)
    UNI_SPARSE = 3    ///< A sparse array (sparse vector or matrix)
};

#define UNI_DATASPEC_MAX_NDIMS 6
typedef struct {
    size_t dims[UNI_DATASPEC_MAX_NDIMS];
    size_t strides[UNI_DATASPEC_MAX_NDIMS];
    size_t nBytes;
    size_t nElems;
    size_t nDims;
    PrimitiveType elemType;
    UniDataKind dataKind;
} UniDataSpec;

typedef UniDataSpec* UniDataSpecDesc;

/**
 * Test the equality of two data specifications.
 */
bool uniDataSpecsEqual(const UniDataSpecDesc d1,
    const UniDataSpecDesc d2);

/**
 * Create a UniDataSpec
 *
 * @param pps   The a pointer to UniDataSpec descriptor, which is to be created.
 */
HPCStatus_t uniCreateDataSpec(UniDataSpecDesc* pps);

/**
 * Destroy a UniDataSpec descriptor and its related memory.
 */
HPCStatus_t uniDestroyDataSpec(UniDataSpecDesc ps);

/**
 * Fill the UniDataSpec for a strided array.
 *
 * @param ps       data specification descriptor.
 * @param ty       The type of the data.
 * @param numDims    The number of dimensions.
 * @param dims     The array of dimension sizes.
 * @param strides  The array of strides along all dimensions (in terms of elements).
 * @return execution status.
 */
HPCStatus_t uniSetDataSpec(PrimitiveType ty,
    size_t numDims,
    const size_t* dims,
    const size_t* strides,
    UniDataSpecDesc ps);

/**
 * Fill the UniDataSpec for a contiguous array.
 *
 * @param ps       data specification descriptor.
 * @param ty       The type of the data.
 * @param numDims    The number of dimensions.
 * @param dims     The array of dimension sizes.
 * @return execution status.
 *
 * @note The strides of the array are determined based on the assumption
 *       that the array is contiguous.
 */
HPCStatus_t uniSetContiguousDataSpec(PrimitiveType ty,
    size_t numDims,
    const size_t* dims,
    UniDataSpecDesc ps);

/**
 * Fill the UniDataSpec for a contiguous 1D array
 *
 * @param ps        data specification descriptor.
 * @param ty        The type of the data.
 * @param len       The length of 1D array.
 * @return execution status.
 *
 * @note    The strides of the array are determined based on the assumption
 *          that the array is contiguous.
 */
HPCStatus_t uniSetContiguous1DataSpec(PrimitiveType ty,
    size_t len,
    UniDataSpecDesc ps);

/**
 * Fill the UniDataSpec for a contiguous 2D array
 *
 * @param ps        A point to the data specification.
 * @param ty        The type of the data.
 * @param width        The size of the 1st dimension.
 * @param height        The size of the 2nd dimension.
 * @return execution status.
 *
 * @note    The strides of the array are determined based on the assumption
 *          that the array is contiguous.
 */
HPCStatus_t uniSetContiguous2DataSpec(PrimitiveType ty,
    size_t width,
    size_t height,
    UniDataSpecDesc ps);

/**
 * Fill the UniDataSpec for a contiguous 3D array
 *
 * @param ps        data specification descriptor.
 * @param ty        The type of the data.
 * @param width        The size of the 1st dimension.
 * @param height        The size of the 2nd dimension.
 * @param channel        The size of the 3th dimension.
 * @return execution status.
 *
 * @note    The strides of the array are determined based on the assumption
 *          that the array is contiguous.
 */
HPCStatus_t uniSetContiguous3DataSpec(
    PrimitiveType ty,
    size_t width,
    size_t height,
    size_t channel,
    UniDataSpecDesc ps);

/**
 * Fill the UniDataSpec for a contiguous 4D array
 *
 * @param ps        data specification descriptor.
 * @param ty        The type of the data.
 * @param width        The size of 1st dimension.
 * @param height        The size of 2nd dimension.
 * @param channel        The size of 3th dimension.
 * @param num        The size of 4th dimension.
 * @return execution status.
 *
 * @note    The strides of the array are determined based on the assumption
 *          that the array is contiguous.
 */
HPCStatus_t uniSetContiguous4DataSpec(PrimitiveType ty,
    size_t width,
    size_t height,
    size_t channel,
    size_t num,
    UniDataSpecDesc ps);

#ifdef __cplusplus
}
#endif

#endif
