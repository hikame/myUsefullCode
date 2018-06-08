#ifndef _OCL_INTERNAL_HEAD_H
#define _OCL_INTERNAL_HEAD_H

#include "oclbin.h"

extern const size_t cl_ave_pool_nchw_len;
extern const uchar  cl_ave_pool_nchw [];

extern const size_t cl_conv_nchw_depthwise_len;
extern const uchar  cl_conv_nchw_depthwise [];

extern const size_t cl_gemm_rowmajor_a_trans_b_notrans_c_notrans_len;
extern const uchar  cl_gemm_rowmajor_a_trans_b_notrans_c_notrans [];

extern const size_t cl_im2col_nchw_len;
extern const uchar  cl_im2col_nchw [];

extern const size_t cl_max_pool_nchw_len;
extern const uchar  cl_max_pool_nchw [];

extern const size_t cl_transpose_2d_len;
extern const uchar  cl_transpose_2d [];

extern const size_t cl_winograd_trans_gather_nchw_len;
extern const uchar  cl_winograd_trans_gather_nchw [];

extern const size_t cl_winograd_trans_scatter_input_nchw_len;
extern const uchar  cl_winograd_trans_scatter_input_nchw [];

#endif
