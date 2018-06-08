#include "internal.h"
#include "oclbin.h"

extern const map<string, clKernel> allOCLKernels = {
  {"cl_ave_pool_nchw", {cl_ave_pool_nchw, cl_ave_pool_nchw_len}},
  {"cl_conv_nchw_depthwise", {cl_conv_nchw_depthwise, cl_conv_nchw_depthwise_len}},
  {"cl_gemm_rowmajor_a_trans_b_notrans_c_notrans", {cl_gemm_rowmajor_a_trans_b_notrans_c_notrans, cl_gemm_rowmajor_a_trans_b_notrans_c_notrans_len}},
  {"cl_im2col_nchw", {cl_im2col_nchw, cl_im2col_nchw_len}},
  {"cl_max_pool_nchw", {cl_max_pool_nchw, cl_max_pool_nchw_len}},
  {"cl_transpose_2d", {cl_transpose_2d, cl_transpose_2d_len}},
  {"cl_winograd_trans_gather_nchw", {cl_winograd_trans_gather_nchw, cl_winograd_trans_gather_nchw_len}},
  {"cl_winograd_trans_scatter_input_nchw", {cl_winograd_trans_scatter_input_nchw, cl_winograd_trans_scatter_input_nchw_len}},
};
