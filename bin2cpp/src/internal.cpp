#include "internal.h"
#include "oclbin.h"

extern const map<string, clKernel> mapOclKernel = {
  {"gemm_image", {gemm_image, gemm_image_len}},
};
