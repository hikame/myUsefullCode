#include "internal.h"
#include "oclbin.h"

extern const map<string, clKernel> allOCLKernels = {
  {"conv_depthwise_nchw", {conv_depthwise_nchw, conv_depthwise_nchw_len}},
  {"mace_conv_2d_3x3", {mace_conv_2d_3x3, mace_conv_2d_3x3_len}},
  {"reduced_reads_5_outputs_armv7", {reduced_reads_5_outputs_armv7, reduced_reads_5_outputs_armv7_len}},
  {"reduced_reads_5_outputs", {reduced_reads_5_outputs, reduced_reads_5_outputs_len}},
  {"snpe", {snpe, snpe_len}},
};
