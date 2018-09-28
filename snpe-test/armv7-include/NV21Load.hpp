#ifndef NV21LOAD_H
#define NV21LOAD_H
#include "SNPE/SNPE.hpp"
#include "DlSystem/ITensorFactory.hpp"

#include "DlSystem/TensorMap.hpp"






std::unique_ptr<zdl::DlSystem::ITensor> loadNV21Tensor (std::unique_ptr<zdl::SNPE::SNPE> & snpe , const char* inputFileListPath);
#endif
