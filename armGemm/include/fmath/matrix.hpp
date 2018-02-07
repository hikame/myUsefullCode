/**
 * @file matrix.hpp
 *
 * @brief matrix related operations
 *
 * @note Only support X86, CUDA and ARM, OCL is not supported now.
 */

#ifndef HPC_FMATH_MATRIX_H_
#define HPC_FMATH_MATRIX_H_

#include <stdio.h>
#include <stdlib.h>
#include "universal/sys.h"

#if defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(USE_OCL)
#include "bcl.h"
#include "bclCommon.hpp"
#include "kernelCache.hpp"
#endif

namespace HPC { namespace fmath {

#if defined(USE_X86)

    //template<typename T>
    //void x86MatrixAxpy(int numCores, size_t numRows, size_t numCols, T a, size_t xcolStride, const T* x,  size_t ycolStride, T* y);

    /**
     * @brief add vector to matrix by row m = m + RowEXT(v)
     * @param v: input 1 x numCols
     * @param m: output numRows x numCols
     */
    template<typename T>
        void x86MatrixAddVectorByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* v, T beta, T* m);


    /**
     * @brief sub vector to matrix by row m = m - RowEXT(v)
     * @param v: input 1 x numCols
     * @param m: output numRows x numCols
     */
    template<typename T>
        void x86MatrixSubVectorByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* v, T beta, T* m);

    /**
     * @brief div vector to matrix by row m = m / RowEXT(v)
     * @param v: input 1 x numCols
     * @param m: output numRows x numCols
     */
    template<typename T>
        void x86MatrixDivideVectorByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* v, T beta, T* m);



    /**
     * @brief add vector to matrix by column m = m + ColEXT(v)
     * @param v: input numRows x 1
     * @param m: output numRows x numCols
     */
    template<typename T>
        void x86MatrixAddVectorByCol(int numCores, size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* v, T beta, T* m);

    /**
     * @brief matrix and vector multiplication r = mv
     */
    template<typename T>
        void x86MatrixGemv_n(int numCores, size_t numRows, size_t numCols,
            T alpha, const T* m, const T* v, T beta, T* r);

    /**
     * @brief sparse matrix and vector multiplication y = Trans(A)x
     * @param transA:         whether A transpose
     * @param csrRowIndex:    sparse row index for A
     * @param csrColIndex:    sparse col indes for A
     */
    template<typename T>
        void x86SparseMatrixGemv(int m, int n,
            int transA,
            const T alpha,
            const int *csrRowIndex, const int *csrColIndex, const T* A,
            const T* x, const T beta, T* y);

    /**
     * @brief matrix and vector multiplication r = Trans(m)v
     * @param numRows: num row for matrix m
     * @param numCols: num col for matrix m
     */
    template<typename T>
        void x86MatrixGemv_t(int numCores, size_t numRows, size_t numCols,
            T alpha, const T* m, const T* v, T beta, T* r);

    /**
     * @brief sparse matrix cross multiplication C = Trans(A)Trans(B)
     * @param transA:       whether A transpose
     * @param transB:       whether B transpose
     * @param csrValA:      sparse value for matrix A
     * @param csrRowIndexA: sparse row index for A
     * @param csrColIndexA: sparse col indes for A
     * TODO
     */
    template<typename T>
        void x86SparseMatrixGemm(int ncores,
            int transA, int transB,
            int m, int n, int k, int nnz, T alpha,
            const int *csrRowIndexA, const int *csrColIndexA, const T *csrValA,
            const T *B, int ldb,
            T beta, T *C, int ldc);

    /**
     * @brief matrix get sum by col v = SumByCol(m)
     * @param numRows: num rows for matrix m and v
     * @param numCols: num cols for matrix m
     */
    template<typename T>
        void x86MatrixGetSum(int numCores, size_t numRows, size_t numCols,
            T alpha, const T* m, T beta, T* v);

    /**
     * @brief matrix get sum by col with stride
     */
    template<typename T>
        void x86MatrixSumByCol(int numCores, size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* m, T beta, T* v);

    /**
     * @brief matrix get sum by row with stride
     */
    template<typename T>
        void x86MatrixSumByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* m, T beta, T* v);

    /**
     * @brief matrix get sum by row with batch
     */
    template<typename T>
        void x86MatrixBatchSumByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, size_t batch, const T alpha, const T* m, const T beta, T* v);

    /**
     * @brief substract max by row with stride
     */
    template<typename T>
        void x86MatrixSubMaxByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, const T* in, T* out);

    /**
     * @brief substract max by row with stride
     */
    template<typename T>
        void x86MatrixSubAverageSumByRow(int NumCores, size_t numRows, size_t numCols,
            size_t stride, const T* in, T* out);

    /**
     * @brief get average by row with stride
     */
    template<typename T>
        void x86MatrixAvgByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, const T alpha, const T* in, const T beta, T* out);

    /**
     * @brief compute probability by row with stride
     */
    template<typename T>
        void x86MatrixProbByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, const T alpha, const T* in, const T beta, T* out);

    /**
     * @brief transpose matrix B = Trans(A)
     */
    template<typename T>
        void x86MatrixTranspose(int numCores, size_t numRows, size_t numCols,
            size_t inColStride, const T* in, size_t outColStride, T* out);

    /**
     * @brief compute max/sum/product from multiple arrays.
     */
    template<typename T>
        void x86Eltwise_max(size_t num, size_t numElements,
            const void* inData[], T* out, const T alpha, const T beta);
    template<typename T>
        void x86Eltwise_sum(size_t num, size_t numElements,
            const void* inData[], T* out, const T* coeff, const T alpha, const T beta);
    template<typename T>
        void x86Eltwise_prod(size_t num, size_t numElements,
            const void* inData[], T* out, const T alpha, const T beta);
    /*
     * @brief compute variance by row.
     */
    template<typename T>
        void x86MatrixGetVarianceByRow(size_t num, size_t dim, size_t stride, T eps, const T *X_EX, T *variance);

    template<typename T>
        void x86MatrixNormalizeVarianceByRow(size_t num, size_t dim, size_t stride, T eps, const T *X_EX, T *out);
#endif //USE_X86

#if defined(USE_ARM)
    /**
     * @brief add vector to matrix by row m = m + RowEXT(v)
     * @param v: input 1 x numCols
     * @param m: output numRows x numCols
     */
    template<typename T>
        void armMatrixAddVectorByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* v, T beta, T* m);

    /**
     * @brief sub vector to matrix by row m = m - RowEXT(v)
     * @param v: input 1 x numCols
     * @param m: output numRows x numCols
     */
    template<typename T>
        void armMatrixSubVectorByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* v, T beta, T* m);

    /**
     * @brief div vector to matrix by row m = m / RowEXT(v)
     * @param v: input 1 x numCols
     * @param m: output numRows x numCols
     */
    template<typename T>
        void armMatrixDivideVectorByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* v, T beta, T* m);


    /**
     * @brief add vector to matrix by column m = m + ColEXT(v)
     * @param v: input numRows x 1
     * @param m: output numRows x numCols
     */
    template<typename T>
        void armMatrixAddVectorByCol(int numCores, size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* v, T beta, T* m);

    /**
     * @brief matrix and vector multiplication r = mv
     */
    template<typename T>
        void armMatrixGemv_n(int numCores, size_t numRows, size_t numCols,
            T alpha, const T* m,  const T* v, T beta, T* r);

    /**
     * @brief the transposed matrix and vector multiplication r = Trans(m)v
     * @param numRows: num row for matrix m
     * @param numCols: num col for matrix m
     */
    template<typename T>
        void armMatrixGemv_t(int numCores, size_t numRows, size_t numCols,
            T alpha, const T* m, const T* v, T beta, T* r);

    /**
     * @brief sparse matrix and vector multiplication y = Trans(A)x
     * @param transA:      whether A transpose
     * @param csrRowIndex: sparse row index for A
     * @param csrColIndex: sparse col indes for A
     */
    template<typename T>
        void armSparseMatrixGemv(int m, int n,
            int transA,
            const T alpha,
            const int *csrRowIndex, const int *csrColIndex, const T* A,
            const T* x, const T beta, T* y);

    /**
     * @brief sparse matrix cross multiplication C = Trans(A)Trans(B)
     * @param transA:      whether A transpose
     * @param transB:      whether B transpose
     * @param csrValA: sparse value for matrix A
     * @param csrRowIndexA: sparse row index for A
     * @param csrColIndexA: sparse col indes for A
     * TODO
     */
    template<typename T>
        void armSparseMatrixGemm(int ncores,
            int transA, int transB,
            int m, int n, int k, int nnz, T alpha,
            const int *csrRowIndexA, const int *csrColIndexA, const T *csrValA,
            const T *B, int ldb,
            T beta, T *C, int ldc);

    /**
     * @brief matrix get sum by col v = SumByCol(m)
     * @param numRows: num rows for matrix m and v
     * @param numCols: num cols for matrix m
     */
    template<typename T>
        void armMatrixGetSum(int numCores, size_t numRows, size_t numCols,
            T alpha, const T* m, T beta, T* v);

    /**
     * @brief matrix get sum by col with stride
     */
    template<typename T>
        void armMatrixSumByCol(int numCores, size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* m, T beta, T* v);

    /**
     * @brief matrix get sum by row with stride
     */
    template<typename T>
        void armMatrixSumByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* m, T beta, T* v);

    /**
     * @brief matrix get sum by row with batch
     */
    template<typename T>
        void armMatrixBatchSumByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, size_t batch, const T alpha, const T* m, const T beta, T* v);

    /**
     * @brief substract max by row with stride
     */
    template<typename T>
        void armMatrixSubMaxByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, const T* in, T* out);
    /**
     * @brief substract max by row with stride
     */
    template<typename T>
        void armMatrixSubAverageSumByRow(int NumCores, size_t numRows, size_t numCols,
            size_t stride, const T* in, T* out);

    /**
     * @brief get average by row with stride
     */
    template<typename T>
        void armMatrixAvgByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, const T alpha, const T* in, const T beta, T* out);

    /**
     * @brief compute probability by row with stride
     */
    template<typename T>
        void armMatrixProbByRow(int numCores, size_t numRows, size_t numCols,
            size_t stride, const T alpha, const T* in, const T beta, T* out);

    /**
     * @brief matrix transpose B = Trans(A)
     */
    template<typename T>
        void armMatrixTranspose(int numCores, size_t numRows, size_t numCols,
            size_t inColStride, const T* in, size_t outColStride, T* out);

    /**
     * @brief compute max/sum/product from multiple arrays.
     */
    template<typename T>
        void armEltwise_max_special(size_t num, size_t numElementsBegin, size_t numElementsEnd,
            const void* inData[], T* out);
    template<typename T>
        void armEltwise_max_normal(size_t num, size_t numElementsBegin, size_t numElementsEnd,
            const void* inData[], T* out, const T alpha, const T beta);
    template<typename T>
        void armEltwise_sum_special(size_t num, size_t numElementsBegin, size_t numElementsEnd,
            const void* inData[], T* out, const T* coeff);
    template<typename T>
        void armEltwise_sum_normal(size_t num, size_t numElementsBegin, size_t numElementsEnd,
            const void* inData[], T* out, const T* coeff, const T alpha, const T beta);
    template<typename T>
        void armEltwise_prod_special(size_t num, size_t numElementsBegin, size_t numElementsEnd,
            const void* inData[], T* out);
    template<typename T>
        void armEltwise_prod_normal(size_t num, size_t numElementsBegin, size_t numElementsEnd,
            const void* inData[], T* out, const T alpha, const T beta);

    /**
     * @brief computevariance by row.
     */
    template<typename T>
        void armMatrixGetVarianceByRow(size_t num, size_t dim, size_t stride, T eps, const T *X_EX, T *variance);

    /**
     * @brief compute normalize variance by row.
     */
    template<typename T>
        void armMatrixNormalizeVarianceByRow(size_t num, size_t dim, size_t stride, T eps, const T *X_EX, T *out);

#endif //USE_ARM

#if defined(USE_CUDA)
    /**
     * @brief add vector to matrix by row m = m + RowEXT(v)
     * @param v: input 1 x numCols
     * @param m: output numRows x numCols
     */
    template<typename T>
        void cudaMatrixAddVectorByRow(int device, cudaStream_t stream, size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* v, T beta, T* m);

    /**
     * @brief add vector to matrix by column m = m + ColEXP(v)
     * @param v: input numRows x 1
     * @param m: output numRows x numCols
     */
    template<typename T>
        void cudaMatrixAddVectorByCol(int device, cudaStream_t stream,
            size_t numRows, size_t numCols,
            size_t stride, T alpha,
            const T* v, T beta, T* m);

    /**
     * @brief matrix additive C = Trans(A) + Trans(B)
     * @param transA: whether A transpose
     * @param transB: whether B transpose
     */
    template<typename T>
        void cudaMatrixGeam(int device, cudaStream_t stream,
            int transA, int transB,
            size_t m, size_t n, T alpha,
            const T* A, size_t lda,
            const T* B, size_t ldb,
            T beta, T* C, size_t ldc);

    /**
     * @brief matrix and vector multiplication r = Trans(m)v
     * @param trans: whether m transpose
     */
    template<typename T>
        void cudaMatrixGemv(int device, cudaStream_t stream,
            int trans, size_t numRows, size_t numCols,
            T alpha, const T* m, size_t ldm, const T* v, T beta, T* r);

    /**
     * @brief sparse matrix cross multiplication y = Trans(A)x
     * @param transA:      whether A transpose
     * @param csrRowIndex: sparse row index for A
     * @param csrColIndex: sparse col indes for A
     */
    template<typename T>
        void cudaSparseMatrixGemv(int device, cudaStream_t stream,
            int transA,
            size_t m, size_t n, size_t numNotZeros,
            const T alpha,
            const int *csrRowIndex, const int *csrColIndex, const T* A,
            const T* x, const T beta, T* y);

    /**
     * @brief matrix cross multiplication C = Trans(A)Trans(B)
     * @param transA:      whether A transpose
     * @param transB:      whether B transpose
     */
    template<typename T>
        void cudaMatrixGemm(int device, cudaStream_t stream,
            int transA, int transB,
            size_t m, size_t n, size_t k, T alpha,
            const T* A, size_t lda,
            const T* B, size_t ldb,
            T beta, T* C, size_t ldc);

    /**
     * @brief sparse matrix cross multiplication C = Trans(A)Trans(B)
     * @param transA:       whether A transpose
     * @param transB:       whether B transpose
     * @param csrValA:      sparse value for matrix A
     * @param csrRowIndexA: sparse row index for A
     * @param csrColIndexA: sparse col indes for A
     * TODO
     */
    template<typename T>
        void cudaSparseMatrixGemm(int device, cudaStream_t stream,
            int transA, int transB,
            size_t m, size_t n, size_t k, size_t nnz, T alpha,
            const int *csrRowIndexA, const int *csrColIndexA, const T *csrValA,
            const T *B, size_t ldb,
            T beta, T *C, size_t ldc);

    /**
     * @brief matrix get sum by col v = SumByCol(m)
     * @param numRows: num rows for matrix m and v
     * @param numCols: num cols for matrix m
     */
    template<typename T>
        void cudaMatrixGetSum(int device, cudaStream_t stream,
            size_t numRows, size_t numCols,
            T alpha, const T* m, T beta, T* v);

    /**
     * @brief matrix get sum by col with stride
     */
    template<typename T>
        void cudaMatrixSumByCol(int device, cudaStream_t stream,
            size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* m, T beta, T* v);

    /**
     * @brief matrix get sum by row with stride
     */
    template<typename T>
        void cudaMatrixSumByRow(int device, cudaStream_t stream,
            size_t numRows, size_t numCols,
            size_t stride, T alpha, const T* m, T beta, T* v);

    /**
     * @brief matrix get sum by row with batch
     */
    template<typename T>
        void cudaMatrixBatchSumByRow(int device, cudaStream_t stream,
            size_t numRows, size_t numCols,
            size_t stride, size_t batch, const T alpha, const T* m, const T beta, T* v);

    /**
     * @brief matrix division by row
     */
    template<typename T>
        void cudaMatrixDivByRow(int device, cudaStream_t stream,
            size_t numRows, size_t numCols,
            T* m, const T* buf);

    /**
     * @brief get max of each row
     */
    template<typename T>
        void cudaMatrixMaxByRow(int device, cudaStream_t stream,
            size_t numRows, size_t numCols,
            const T* in, T* out);

    /**
     * @brief get max value and conresponding index of each row
     */
    template<typename T>
        void cudaMatrixMaxByRowIndex(int device, cudaStream_t stream,
            int numRows, int numCols,
            int *index, const T* in, T* out);

    /**
     * @brief matrix subtraction by row
     */
    template<typename T>
        void cudaMatrixSubByRow(int device, cudaStream_t stream,
            size_t numRows, int numCols,
            const T* in, const T* rowData, T* out);

    /**
     * @brief matrix transpose B = Trans(A)
     */
    template<typename T>
        void cudaMatrixTranspose(int device, cudaStream_t stream,
            size_t numRows, size_t numCols,
            size_t inColStride, const T* in, size_t outColStride, T* out);
    /**
     * @brief substract max by row with stride
     */
    template<typename T>
        void cudaMatrixSubAverageSumByRow(int device, cudaStream_t stream, size_t numRows, size_t numCols,
            size_t stride, const T* in, T* out);

    template<typename T>
        void cudaMatrixNormalizeVarianceByRow(int device, cudaStream_t stream,
            size_t num, size_t dim, size_t stride, T eps, const T *X_EX, T *out);
#endif //USE_CUDA

#if defined(USE_OCL)
    template<typename T>
        void oclMatrixGemv_spe_p1(cl_command_queue queue, KernelCache* kc, int in_w, int in_h,
            T alpha, T beta, cl_mem in_matrix, cl_mem in_vector, cl_mem out_vector, bool enableImg);
    template<typename T>
        void oclMatrixGemv_spe_p2(cl_command_queue queue, KernelCache* kc, int in_w, int in_h, int has_bias, int out_w,
            cl_mem bias, cl_mem in, cl_mem out);
    template<typename T>
        void oclMatrixGeam(cl_command_queue queue, KernelCache* kc,
                int transA, int transB,
                cl_uint m, cl_uint n, T alpha,
                cl_mem A,  cl_uint lda,
                cl_mem B,  cl_uint ldb,
                T beta, cl_mem C, cl_uint ldc);

    template<typename T>
        void oclMatrixAddVectorByRow(cl_command_queue queue, KernelCache* kc, cl_uint numRows, cl_uint numCols,
                cl_uint stride, T alpha, const cl_mem v, T beta, cl_mem m);

    template<typename T>
        void oclMatrixAddVectorByCol(cl_command_queue queue, KernelCache* kc, cl_uint numRows, cl_uint numCols,
                cl_uint stride, T alpha, const cl_mem v, T beta, cl_mem m);


    template<typename T>
        void oclMatrixCAZP(cl_command_queue queue, KernelCache* kc,
        cl_uint offRowSrc, cl_uint numRowsSrc, cl_uint offColSrc, cl_uint numColsSrc, cl_mem src,
        cl_uint offRowDst, cl_uint numRowsDst, cl_uint offColDst, cl_uint numColsDst, cl_mem dst, cl_uint group);


    template<typename T>
        void oclSparseGemv_n(cl_command_queue queue, int m, int n, int numNotZeros,
                T alpha,
                const cl_mem csrRowIndex, const cl_mem csrColIndex, const cl_mem A,
                const cl_mem x, T beta, cl_mem y);

    template<typename T>
        void oclSparseGemv_t(cl_command_queue queue,
                int m, int n, int numNotZeros,
                T alpha,
                const cl_mem csrRowIndex, const cl_mem csrColIndex, const cl_mem A,
                const cl_mem x, T beta, cl_mem y);
#endif //USE_OCL

#if defined(USE_X86)
    template<typename T>
        size_t x86GemmNNGetSize(int numCores, size_t m, size_t n, size_t k,
            size_t lda, size_t ldb, size_t ldc);
    template<typename T>
        size_t x86GemmNTGetSize(int numCores, size_t m, size_t n, size_t k,
            size_t lda, size_t ldb, size_t ldc);
    template<typename T>
        size_t x86GemmTNGetSize(int numCores, size_t m, size_t n, size_t k,
            size_t lda, size_t ldb, size_t ldc);
    template<typename T>
        size_t x86GemmTTGetSize(int numCores, size_t m, size_t n, size_t k,
            size_t lda, size_t ldb, size_t ldc);
    /**
     *@{
     */
    /**
     * @brief gemm: matrix multiplication
     */
    template<typename T>
        void x86Gemm_nn(int numCores, size_t m, size_t n, size_t k,
            T alpha, size_t lda, const T* A, size_t ldb, const T* B,
            T *tmpBuffer, T beta, size_t ldc, T* C);

    template<typename T>
        void x86Gemm_nt(int numCores, size_t m, size_t n, size_t k,
            T alpha, size_t lda, const T* A, size_t ldb, const T* B,
            T *tempBuffer, T beta, size_t ldc, T* C);

    template<typename T>
        void x86Gemm_tn(int numCores, size_t m, size_t n, size_t k,
            T alpha, size_t lda, const T* A, size_t ldb, const T* B,
            T *tmpBuffer, T beta, size_t ldc, T* C);

    template<typename T>
        void x86Gemm_tt(int numCores, size_t m, size_t n, size_t k,
            T alpha, size_t lda, const T* A, size_t ldb, const T* B,
            T *tmpBuffer, T beta, size_t ldc, T* C);
    /**
     * @}
     */

    /**
     *@{
     */
    /**
     * @brief geam: matrix additive
     */
    template<typename T>
        void x86Geam_nn(int numCores, size_t m, size_t n,
            T alpha, size_t lda, const T* A,
            T beta, size_t ldb, const T* B, size_t ldc, T* C);

    template<typename T>
        void x86Geam_nt(int numCores, size_t m, size_t n,
            T alpha, size_t lda, const T* A,
            T beta, size_t ldb, const T* B, size_t ldc, T* C);

    template<typename T>
        void x86Geam_tn(int numCores, size_t m, size_t n,
            T alpha, size_t lda, const T* A,
            T beta, size_t ldb, const T* B, size_t ldc, T* C);

    template<typename T>
        void x86Geam_tt(int numCores, size_t m, size_t n,
            T alpha, size_t lda, const T* A,
            T beta, size_t ldb, const T* B, size_t ldc, T* C);
    /**
     * @}
     */
#endif //USE_X86

#if defined(USE_ARM)
    template<typename T>
        size_t armGemmNNGetSize(int numCores, size_t m, size_t n, size_t k,
            size_t lda, size_t ldb, size_t ldc);
    template<typename T>
        size_t armGemmNTGetSize(int numCores, size_t m, size_t n, size_t k,
            size_t lda, size_t ldb, size_t ldc);
    template<typename T>
        size_t armGemmTNGetSize(int numCores, size_t m, size_t n, size_t k,
            size_t lda, size_t ldb, size_t ldc);
    template<typename T>
        size_t armGemmTTGetSize(int numCores, size_t m, size_t n, size_t k,
            size_t lda, size_t ldb, size_t ldc);

    /**
     * @{
     */
    /**
     * @brief gemm: matrix multiplication
     */
    template<typename T>
        void armGemm_nn(int numCores, size_t m, size_t n, size_t k,
            T alpha, size_t lda, const T* A, size_t ldb, const T* B,
            T *temp, T beta, size_t ldc, T* C);

    template<typename T>
        void armGemm_nt(int numCores, size_t m, size_t n, size_t k,
            T alpha, size_t lda, const T* A, size_t ldb, const T* B,
            T *temp, T beta, size_t ldc, T* C);

    template<typename T>
        void armGemm_tn(int numCores, size_t m, size_t n, size_t k,
            T alpha, size_t lda, const T* A, size_t ldb, const T* B,
            T *temp, T beta, size_t ldc, T* C);

    template<typename T>
        void armGemm_tt(int numCores, size_t m, size_t n, size_t k,
            T alpha, size_t lda, const T* A, size_t ldb, const T* B,
            T *temp, T beta, size_t ldc, T* C);
    /**
     * @}
     */

    //@{
    /**
     * @brief geam: matrix additive
     */
    template<typename T>
        void armGeam_nn(int numCores, size_t m, size_t n,
            T alpha, size_t lda, const T* A,
            T beta, size_t ldb, const T* B, size_t ldc, T* C);

    template<typename T>
        void armGeam_nt(int numCores, size_t m, size_t n,
            T alpha, size_t lda, const T* A,
            T beta, size_t ldb, const T* B, size_t ldc, T* C);

    template<typename T>
        void armGeam_tn(int numCores, size_t m, size_t n,
            T alpha, size_t lda, const T* A,
            T beta, size_t ldb, const T* B, size_t ldc, T* C);

    template<typename T>
        void armGeam_tt(int numCores, size_t m, size_t n,
            T alpha, size_t lda, const T* A,
            T beta, size_t ldb, const T* B, size_t ldc, T* C);
    /**
     * @}
     */
#endif //USE_ARM

#if defined(USE_CUDA)
    /**
     * @{
     */
    /**
     * @brief gemm: matrix multiplication
     */
    template<typename T>
        void cudaGemm_nn(int device, cudaStream_t stream,
            size_t m, size_t n, size_t k, size_t lda, const T* A,
            size_t ldb, const T* B, size_t ldc, T* C);

    template<typename T>
        void cudaGemm_nt(int device, cudaStream_t stream,
            size_t m, size_t n, size_t k, size_t lda, const T* A,
            size_t ldb, const T* B, size_t ldc, T* C);

    template<typename T>
        void cudaGemm_tn(int device, cudaStream_t stream,
            size_t m, size_t n, size_t k, size_t lda, const T* A,
            size_t ldb, const T* B, size_t ldc, T* C);

    template<typename T>
        void cudaGemm_tt(int device, cudaStream_t stream,
            size_t m, size_t n, size_t k, size_t lda, const T* A,
            size_t ldb, const T* B, size_t ldc, T* C);
    /**
     * @}
     */
#endif //USE_CUDA

#if defined(USE_OCL)
    /**
     * @{
     */
    /**
     * @brief gemm: matrix multiplication
     */
    //TODO

    /**
     * @brief matrix transpose B = Trans(A)
     */
    template<typename T>
        void oclMatrixTranspose(cl_command_queue queue, KernelCache* kc,
                cl_uint numRows, cl_uint numCols, cl_uint in_off,
                cl_uint inColStride, cl_mem in, cl_uint outColStride, cl_mem out);

    /**
     * @brief matrix transpose B = Trans(A), and B will auto complete the empty elems with zero
     */
    template<typename T>
        void oclMatrixTransposeAndComplete(cl_command_queue queue, KernelCache* kc,
                cl_uint widthIn, cl_uint heightIn, cl_uint widthOut, cl_uint heightOut,
                cl_uint group, cl_mem in, cl_mem out);

    template<typename T>
        void oclMatrixSubAverageSumByRow(cl_command_queue queue, KernelCache* kc, cl_uint numRows, cl_uint numCols,
                cl_uint stride, cl_mem in, cl_mem out);
    template<typename T>
        void oclMatrixNormalizeVarianceByRow(cl_command_queue queue, KernelCache* kc, cl_uint row, cl_uint col, cl_uint channel, cl_uint prow, cl_uint pcol,
            cl_uint padding, cl_uint shift, T eps, cl_mem in, cl_mem out, bool USE_IMG);

    /**
     * @}
     */
    //@}

    template<typename T>
        void copyBufferToImage(cl_command_queue queue, KernelCache* kc, size_t width, size_t height, size_t depth, cl_mem buffer, cl_mem image);

    template<typename T>
        void allocImageAndCopy(cl_command_queue queue, KernelCache* kc, cl_mem buffer, cl_mem * image, size_t width, size_t height);

    template<typename T>
        void allocImage(cl_command_queue queue, cl_mem *image, size_t width, size_t height, bool readonly = false);

    template<typename T>
        void allocImage1d(cl_command_queue queue, cl_mem *image, size_t len, bool readonly = false);

    template<typename T>
        void allocImage3d(cl_command_queue queue, cl_mem *image, size_t width, size_t height, size_t depth, bool readonly = false);

#endif //USE_OCL

} } //end namespace fmath //end namespace HPC

#endif //HPC_FMATH_MATRIX_H_
