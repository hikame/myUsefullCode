#ifndef LOADINPUTTENSOR_H
#define LOADINPUTTENSOR_H
#include "SNPE/SNPE.hpp"
#include "DlSystem/ITensorFactory.hpp"

#include "DlSystem/TensorMap.hpp"





std::unique_ptr<zdl::DlSystem::ITensor> loadInputTensor (std::unique_ptr<zdl::SNPE::SNPE> & snpe , std::string& fileLine);
zdl::DlSystem::TensorMap loadMultipleInput (std::unique_ptr<zdl::SNPE::SNPE> & snpe , std::string& fileLine);

#endif
