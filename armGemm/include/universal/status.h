#ifndef HPC_UNIVERSAL_STATUS_H
#define HPC_UNIVERSAL_STATUS_H
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * Function return status in universal
     */
    typedef enum {
        // basic status
        HPC_SUCCESS              = 0,   ///< The function completes successfully
        HPC_NOT_SUPPORTED        = 1,   ///< The functionality is not supported.
        HPC_NOT_IMPLEMENTED      = 2,   ///< The functionality has not been implemented yet.
        HPC_UNEXPECTED_NULL      = 3,   ///< Encountered an unexpected null pointer.
        HPC_UNEXPECTED_OP        = 4,   ///< Unexpected operation or an operation unexpected in certain place

        // related to arguments
        HPC_INVALID_DEVICE       = 5,   ///< invalid device.
        HPC_POINTER_NULL         = 6,   ///< null pointer.
        HPC_INVALID_ARGS         = 7,   ///< The values of the input arguments were invalid.
        HPC_OUT_OF_BOUND         = 8,   ///< The input indexes were out of bound.
        HPC_DIMS_MISMATCHED      = 9,   ///< The dimensions of input arrays are inconsistent.

        // failures of operations
        HPC_OP_NOT_PERMITED      = 10,   ///< operation is not permited
        HPC_ALLOC_FAILED         = 11,   ///< Memory allocation was failed.
        HPC_EXECUTION_FAILED     = 12,  ///< Execution of certain operations was failed.
        HPC_IOERROR              = 13,  ///< Errors related to file input/output.

        // conflicts with current states
        HPC_NOT_INITIALIZED      = 14,  ///< The data/value was not initialized
        HPC_ALREADY_INITIALIZED  = 15,  ///< The data/value was already initialized (attempted to initialize twice)
        HPC_KEY_NOT_FOUND        = 16,  ///< Attempted to use a key which does not exist.
        HPC_DUPLICATED_KEY       = 17,  ///< Attempted to insert a key that already existed.

        // misc
        HPC_PARSE_ERROR          = 18,  ///< error arising during parsing
        HPC_ILLDEFINED           = 19,  ///< ill-defined task or problem

        // others
        HPC_OTHER_ERROR          = 255  ///< Errors that do not belong to any of the classes above.
    } HPCStatus_t;

    /**
     * Get human-readable message from return status
     *
     * @param hst   Function return status.
     *
     * @note    The function returns an empty string when the input status code
     *          does not match any of the predefined enumerators.
     */
    inline const char* uniGetStatusString(HPCStatus_t hst) {
        switch (hst) {
            case HPC_SUCCESS:
                return "Success";
            case HPC_NOT_SUPPORTED:
                return "Not supported";
            case HPC_NOT_IMPLEMENTED:
                return "Not implemented";
            case HPC_UNEXPECTED_NULL:
                return "Unexpected null pointer";
            case HPC_UNEXPECTED_OP:
                return "Unexpected operation";

            case HPC_INVALID_DEVICE:
                return "Invalid devices";
            case HPC_POINTER_NULL:
                return "Unexpected null pointer";
            case HPC_INVALID_ARGS:
                return "Invalid arguments";
            case HPC_OUT_OF_BOUND:
                return "Index out of bound";
            case HPC_DIMS_MISMATCHED:
                return "Mismatched dimensions of array arguments";

            case HPC_OP_NOT_PERMITED:
                return "Operation not permited";
            case HPC_ALLOC_FAILED:
                return "Failed memory allocation";
            case HPC_EXECUTION_FAILED:
                return "Failed execution";
            case HPC_IOERROR:
                return "File I/O error";

            case HPC_NOT_INITIALIZED:
                return "Not initialized";
            case HPC_ALREADY_INITIALIZED:
                return "Already initialized";
            case HPC_KEY_NOT_FOUND:
                return "Key not found";
            case HPC_DUPLICATED_KEY:
                return "Duplicated key";

            case HPC_PARSE_ERROR:
                return "Parse error";
            case HPC_ILLDEFINED:
                return "Ill-defined problem/task";

            case HPC_OTHER_ERROR:
                return "Other unknown error";
            default:
                return "";
        }
    }

#define printMessage(msg) {printf("%s %d %s\n", __FILE__, __LINE__, msg); fflush(stdout);}

#ifdef __cplusplus
}
#endif

#if defined(USE_CUDA)
#include <curand.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <driver_types.h>

/**
 * @brief used to map cuda error to HPC Status
 * that give us the opportunity to return HPCStatus_t for cuda error
 **/
inline HPCStatus_t mapCudaErrorToHPCStatus(cudaError_t cet) {
    switch(cet){
    case cudaSuccess :
        return HPC_SUCCESS;
    case cudaErrorInitializationError :
        return HPC_NOT_INITIALIZED;
    case cudaErrorMemoryAllocation :
        return HPC_ALLOC_FAILED;
    case cudaErrorNotYetImplemented :
        return HPC_NOT_IMPLEMENTED;
    case cudaErrorLaunchFailure :
        return HPC_EXECUTION_FAILED;
    case cudaErrorInvalidValue :
        return HPC_INVALID_ARGS;
    default :
        return HPC_OTHER_ERROR;
    }
}

/**
 * @brief used to map cublas status to HPC Status
 * that give us the opportunity to return HPCStatus_t for cublas error
 **/
inline HPCStatus_t mapCublasStatusToHPCStatus(cublasStatus_t cst) {
    switch(cst){
    case CUBLAS_STATUS_SUCCESS :
        return HPC_SUCCESS;
    case CUBLAS_STATUS_NOT_INITIALIZED :
        return HPC_NOT_INITIALIZED;
    case CUBLAS_STATUS_ALLOC_FAILED :
        return HPC_ALLOC_FAILED;
    case CUBLAS_STATUS_INVALID_VALUE :
        return HPC_INVALID_ARGS;
    case CUBLAS_STATUS_EXECUTION_FAILED :
        return HPC_EXECUTION_FAILED;
    case CUBLAS_STATUS_NOT_SUPPORTED :
        return HPC_NOT_SUPPORTED;
    default :
        return HPC_OTHER_ERROR;
    }
}

/**
 * @brief used to map cusparse status to HPC Status
 * that give us the opportunity to return HPCStatus_t for curand error
 **/
inline HPCStatus_t mapCusparseStatusToHPCStatus(cusparseStatus_t cst) {
    switch(cst){
    case CUSPARSE_STATUS_SUCCESS :
        return HPC_SUCCESS;
    case CUSPARSE_STATUS_NOT_INITIALIZED :
        return HPC_NOT_INITIALIZED;
    case CUSPARSE_STATUS_ALLOC_FAILED :
        return HPC_ALLOC_FAILED;
    case CUSPARSE_STATUS_INVALID_VALUE :
        return HPC_INVALID_ARGS;
    case CUSPARSE_STATUS_EXECUTION_FAILED :
        return HPC_EXECUTION_FAILED;
    default :
        return HPC_OTHER_ERROR;
    }
}

/**
 * @brief used to map curand status to HPC Status
 * that give us the opportunity to return HPCStatus_t for curand error
 **/
inline HPCStatus_t mapCurandStatusToHPCStatus(curandStatus_t cet) {
    switch(cet){
    case CURAND_STATUS_SUCCESS :
        return HPC_SUCCESS;
    case CURAND_STATUS_NOT_INITIALIZED :
        return HPC_NOT_INITIALIZED;
    case CURAND_STATUS_ALLOCATION_FAILED :
        return HPC_ALLOC_FAILED;
    case CURAND_STATUS_LAUNCH_FAILURE :
        return HPC_EXECUTION_FAILED;
    default :
        return HPC_OTHER_ERROR;
    }
}
#endif

#endif
