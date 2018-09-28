#include <iostream>
#include <algorithm>
#include "SNPE/SNPE.hpp"
#include <sstream>
#include "Util.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/TensorShape.hpp"

void executeNetwork (std::unique_ptr<zdl::SNPE::SNPE> & snpe , std::unique_ptr<zdl::DlSystem::ITensor> & input, std::string OutputDir, int num)

{
    //Execute the network and store the outputs that were specified when creating the network in a TensorMap
    static zdl::DlSystem::TensorMap outputTensorMap;
    snpe->execute(input.get(), outputTensorMap);
    zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();

    //Iterate through the output Tensor map, and print each output layer name
    std::for_each( tensorNames.begin(), tensorNames.end(), [&](const char* name)
    {
        std::ostringstream path;
        path << OutputDir << "/"
        << "Result_" << num << "/"
        << name << ".raw";
        auto tensorPtr = outputTensorMap.getTensor(name);
        SaveITensor(path.str(), tensorPtr);
    });
}
