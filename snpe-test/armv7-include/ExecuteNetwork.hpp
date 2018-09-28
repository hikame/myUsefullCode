#ifndef EXECUTE_H
#define EXECUTE_H
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/TensorMap.hpp"
#include "Util.hpp"






void executeNetwork (std::unique_ptr<zdl::SNPE::SNPE> & snpe , std::unique_ptr<zdl::DlSystem::ITensor> & input, std::string OutputDir, int num);

#endif
