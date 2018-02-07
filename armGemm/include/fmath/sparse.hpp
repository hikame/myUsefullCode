/**
 * @file sparse.hpp
 *
 * @brief sparse matrix related operations
 *
 * @note Only support X86 and ARM; CUDA and OCL is not supported now.
 */

#ifndef HPC_FMATH_SPARSE_H_
#define HPC_FMATH_SPARSE_H_

#include <stdio.h>
#include <stdlib.h>
#include "universal/sys.h"

#if defined(USE_OCL)
#include "bcl.h"
#endif

namespace HPC { namespace fmath {

#if defined(USE_X86)||defined(USE_ARM)||defined(CUDA)

    /*****************************************************************************
      get the num of Bytes of the space which a matrix pointer refer to in matrix format transform
     *****************************************************************************/
    //get the num of Bytes of the space which a matrix pointer refer to in tranfering dense matrix to csr fromat
    template<typename T>
        void sparse_matrix_dense2csr_getSize(int numRows, int numCols, const T *memory, size_t &non_zeros, size_t &rowStart_size, size_t &colIndex_size, size_t &vals_size);

    //get the num of Bytes of the space which a matrix pointer refer to in tranfering dense matrix to bcsr fromat
    template<typename T>
        void sparse_matrix_dense2bcsr_getSize(int numRows, int numCols, const T *memory, int block_size, size_t &non_zeros,
            size_t &rowId_size, size_t &rowInfo_size, size_t &colIndex_size, size_t &vals_size);

    //get the num of Bytes of the space which a matrix pointer refer to in tranfering dense matrix to ccs fromat
    template<typename T>
        void sparse_matrix_dense2ccs_getSize(int numRows, int numCols, const T *memory, size_t &non_zeros, size_t &colStart_size, size_t &rowIndex_size, size_t &vals_size);

    //get the num of Bytes of the space which a matrix pointer refer to in tranfering dense matrix to bccs fromat
    template<typename T>
        void sparse_matrix_dense2bccs_getSize(int numRows, int numCols, const T *memory, int block_size, size_t &non_zeros,
            size_t &colId_size, size_t &colInfo_size, size_t &rowIndex_size, size_t &vals_size);

    //get the num of Bytes of the space which a matrix pointer refer to in tranfering coo format to csr fromat
    template<typename T>
        void sparse_matrix_coo2csr_getSize(int numRows, int numCols, size_t non_zeros, size_t &csr_rowStart_size, size_t &csr_colIndex_size, size_t &csr_vals_size);

    //get the num of Bytes of the space which a matrix pointer refer to in tranfering coo format to bcsr fromat
    template<typename T>
        void sparse_matrix_coo2bcsr_getSize(int numRows, int numCols, size_t non_zeros, int block_size,
            size_t &bcsr_rowId_size, size_t &bcsr_rowInfo_size, size_t &bcsr_colIndex_size, size_t &bcsr_vals_size);

    //get the num of Bytes of the space which a matrix pointer refer to in tranfering coo format to ccs fromat
    template<typename T>
        void sparse_matrix_coo2ccs_getSize(int numRows, int numCols, size_t non_zeros, size_t &ccs_colStart_size, size_t &ccs_rowIndex_size, size_t &ccs_vals_size);

    //get the num of Bytes of the space which a matrix pointer refer to in tranfering coo format to bccs fromat
    template<typename T>
        void sparse_matrix_coo2bccs_getSize(int numRows, int numCols, size_t non_zeros, int block_size,
            size_t &bccs_colId_size, size_t &bccs_colInfo_size, size_t &bccs_rowIndex_size, size_t &bccs_vals_size);


    /*****************************************************************************
      the format transformation of all kinds of sparse matrix format
     *****************************************************************************/

    //transfer denseMat with lots of zeros to csr format;
    template<typename T>
        void sparse_matrix_dense2csr(int numRows, int numCols, const T *memory, int* rowStart, int* colIndex, T* vals);

    //transfer denseMat with lots of zeros to bcsr format;
    template<typename T>
        void sparse_matrix_dense2bcsr(int numRows, int numCols, const T *memory, int block_size,
            int *rowId, unsigned short* rowInfo, unsigned short* colIndex, T* vals);

    //transfer denseMat with lots of zeros to ccs format;
    template<typename T>
        void sparse_matrix_dense2ccs(int numRows, int numCols, const T *memory, int* colStart, int* rowIndex, T* vals);

    //transfer denseMat with lots of zeros to bccs format;
    template<typename T>
        void sparse_matrix_dense2bccs(int numRows, int numCols, const T *memory, int block_size,
            int *colId, unsigned short* colInfo, unsigned short* rowIndex, T* vals);

    //transfer coo format to csr format;
    template<typename T>
        void sparse_matrix_coo2csr(int numRows, int numCols, size_t non_zeros, const int *coo_rowIndex, const int *coo_colIndex, const T *coo_vals,
            int* csr_rowStart, int* csr_colIndex, T* csr_vals);

    //transfer coo format to bcsr format;
    template<typename T>
        void sparse_matrix_coo2bcsr(int numRows, int numCols, size_t non_zeros, const int *coo_rowIndex, const int *coo_colIndex, const T *coo_vals,
            int block_size, int *bcsr_rowId, unsigned short* bcsr_rowInfo, unsigned short* bcsr_colIndex, T* bcsr_vals);

    //transfer coo format to ccs format;
    template<typename T>
        void sparse_matrix_coo2ccs(int numRows, int numCols, size_t non_zeros, const int *coo_rowIndex, const int *coo_colIndex, const T *coo_vals,
            int* ccs_colStart, int* ccs_rowIndex, T* ccs_vals);

    //transfer coo format to bccs format;
    template<typename T>
        void sparse_matrix_coo2bccs(int numRows, int numCols, size_t non_zeros, const int *coo_rowIndex, const int *coo_colIndex, const T *coo_vals,
            int block_size, int *bccs_colId, unsigned short* bccs_colInfo, unsigned short* bccs_rowIndex, T* bccs_vals);
#endif

    /*****************************************************************************
      the sparse matrix dense vector multiplication of all kinds of sparse matrix format
     *****************************************************************************/
#if defined(USE_X86)
    //csr format sparse matrix dense vector multiplication
    template<typename T>
        void x86Spmv_csr(int numCores, int numRows,
            const int* rowStart, const int* colIndex, const T* A, const T* x, T* y);

    //bcsr firmat sparse matrix dense vector multiplication
    template<typename T>
        void x86Spmv_bcsr(int numCores, int numRows, int numcols, int block_size,
            const int *rowId, const unsigned short* rowInfo, const unsigned short* colIndex, const T* A, const T* x, T* y);

    //ccs firmat sparse matrix dense vector multiplication
    template<typename T>
        void x86Spmv_ccs(int numCores, int numRows, int numCols,
            const int* colStart, const int* rowIndex, const T* A, const T* x, T* y);

    //bccs format sparse matrix dense vector multiplication
    template<typename T>
        void x86Spmv_bccs(int numCores, int numRows, int numcols, int block_size,
            const int *colId, const unsigned short* colInfo, const unsigned short* rowIndex, const T* A, const T* x, T* y);
#endif

#if defined(USE_ARM)
    //csr format sparse matrix dense vector multiplication
    template<typename T>
        void armSpmv_csr(int numCores, int numRows, const int* rowStart, const int* colIndex, const T* A, const T* x, T* y);

    //bcsr firmat sparse matrix dense vector multiplication
    template<typename T>
        void armSpmv_bcsr(int numCores, int numRows, int numcols, int block_size,
            const int *rowId, const unsigned short* rowInfo, const unsigned short* colIndex, const T* A, const T* x, T* y);

    //ccs firmat sparse matrix dense vector multiplication
    template<typename T>
        void armSpmv_ccs(int numCores, int numCols, const int* colStart, const int* rowIndex, const T* A, const T* x, T* y);

    //bccs format sparse matrix dense vector multiplication
    template<typename T>
        void armSpmv_bccs(int numCores, int numRows, int numcols, int block_size,
            const int *colId, const unsigned short* colInfo, const unsigned short* rowIndex, const T* A, const T* x, T* y);
#endif

    /*****************************************************************************
      the sparse matrix dense matrix multiplication of all kinds of sparse matrix format
     *****************************************************************************/
#if defined(USE_X86)
    /**
     * @brief csr format sparse matrix dense matrix multiplication
     @sparse matrix m*k multiplicate k*n
     */
    template<typename T>
        void x86Spmm_csr(int numCores, int m, int n, int k,
            const int* rowStart, const int* colIndex, const T* A, const T* B, int ldb, T* C, int ldc);

    /**
     * @brief bcsr format sparse matrix dense matrix multiplication
     @sparse matrix m*k multiplicate k*n
     */
    template<typename T>
        void x86Spmm_bcsr(int numCores, int m, int n, int k, int block_size, const int *rowId,
            const unsigned short* rowInfo, const unsigned short* colIndex, const T* A, const T* B, int ldb, T* C, int ldc);

    /**
     * @brief ccs format sparse matrix dense matrix multiplication
     @sparse matrix m*k multiplicate k*n
     */

    template<typename T>
        void x86Spmm_ccs(int numCores, int m, int n, int k,
            const int* colStart, const int* rowIndex, const T* A, const T* B, int ldb, T* C, int ldc);

    /**
     * @brief bccs format sparse matrix dense matrix multiplication
     @sparse matrix m*k multiplicate k*n
     */

    template<typename T>
        void x86Spmm_bccs(int numCores, int m, int n, int k, int block_size, const int *colId,
            const unsigned short* colInfo, const unsigned short* rowIndex, const T* A, const T* B, int ldb, T* C, int ldc);
#endif

#if defined(USE_ARM)
    /**
     * @brief csr format sparse matrix dense matrix multiplication
     @sparse matrix m*k multiplicate k*n
     */
    template<typename T>
        void armSpmm_csr(int numCores, int m, int n, int k,
            const int* rowStart, const int* colIndex, const T* A, const T* B,  int ldb, T* C, int ldc);

    /**
     * @brief bcsr format sparse matrix dense matrix multiplication
     @sparse matrix m*k multiplicate k*n
     */
    template<typename T>
        void armSpmm_bcsr(int numCores, int m, int n, int k, int block_size, const int *rowId,
            const unsigned short* rowInfo, const unsigned short* colIndex, const T* A, const T* B, int ldb, T* C, int ldc);

    /**
     * @brief ccs format sparse matrix dense matrix multiplication
     @sparse matrix m*k multiplicate k*n
     */
    template<typename T>
        void armSpmm_ccs(int numCores, int m, int n, int k,
            const int* colStart, const int* rowIndex, const T* A, const T* B, int ldb, T* C, int ldc);

    /**
     * @brief bccs format sparse matrix dense matrix multiplication
     @sparse matrix m*k multiplicate k*n
     */
    template<typename T>
        void armSpmm_bccs(int numCores, int m, int n, int k, int block_size, const int *colId,
            const unsigned short* colInfo, const unsigned short* rowIndex, const T* A, const T* B,  int ldb, T* C, int ldc);
#endif

#if defined(USE_X86)||defined(USE_ARM)
    /***************************************************************************
      the interface used for multi-thread
     **************************************************************************/
    /**
      notice： the inner interface only give the stategy for load balance, the
      detail need user to implement with certain data stucture.
     **/
    //get the num of Bytes of the space which the spilt index refer to
    void sparse_matrix_split_idx_getSize(int core, size_t &split_idx_size);

    /**
     * @brief split the dense matrix to number cores thread for load balance.
     @param dir: the split direction. if dir == 0, split by the horizotal(row), else
     if dir == 1, split by the vertical(col).
     @split_idx: the split index, the thread[i] gets the row(col) index of [split_idx[i], split_idx[i+1]).
     */
    template<typename T>
        void sparse_matrix_split_dense_lb_nz(int numRows, int numCols, const T *memory, int cores, int dir, int *split_idx);

    /**
     * @brief split the coo matrix to number cores thread for laod balance.
     @param rowIndex_sort, col_index_sort, vals_sort: the coo array after sort, if split by horizatal, sort as rowMajor,
     if split by vertical, sort as colMajor.
     @param dir: the split direction. if dir == 0, split by the horizotal(row), else
     if dir == 1, split by the vertical(col).
     @split_idx: the split index, the thread[i] gets the row(col) index of [split_idx[i], split_idx[i+1]).
     */
    template<typename T>
        void sparse_matrix_split_coo_lb_nz(int numRows, int numCols, size_t non_zeros, const int *rowIndex, const int *colIndex, const T *vals, int core, int dir,
            int *rowIndex_sort, int *colIndex_sort, T *vals_sort, int *split_idx);

    /**
     * @brief split the csr matrix to number cores thread for laod balance.
     @param dir: the split direction. if dir == 0, split by the horizotal(row), else
     if dir == 1, split by the vertical(col).
     @split_idx: the split index, the thread[i] gets the row(col) index of [split_idx[i], split_idx[i+1]).
     */
    template<typename T>
        void sparse_matrix_split_csr_lb_nz(int numRows, int numCols, size_t non_zeros, const int *rowStart, const int *colIndex, const T *vals, int core, int dir, int *split_idx);

    /**
     * @brief split the ccs matrix to number cores thread for laod balance.
     @param dir: the split direction. if dir == 0, split by the horizotal(row), else
     if dir == 1, split by the vertical(col).
     @split_idx: the split index, the thread[i] gets the row(col) index of [split_idx[i], split_idx[i+1]).
     */
    template<typename T>
        void sparse_matrix_split_ccs_lb_nz(int numRows, int numCols, size_t non_zeros, const int *colStart, const int *rowIndex, const T *vals, int core, int dir, int *split_idx);
#endif
}  //end namespace fmath
}; //end namespace HPC

#endif //HPC_FMATH_SPARSE_H_
