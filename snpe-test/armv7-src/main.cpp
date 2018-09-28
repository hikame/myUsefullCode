//==============================================================================
//
//  @@
//
//  Copyright 2015-2017 Qualcomm Technologies, Inc. All rights reserved.
//  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
//
//  The party receiving this software directly from QTI (the "Recipient")
//  may use this software as reasonably necessary solely for the purposes
//  set forth in the agreement between the Recipient and QTI (the
//  "Agreement"). The software may be used in source code form solely by
//  the Recipient's employees (if any) authorized by the Agreement. Unless
//  expressly authorized in the Agreement, the Recipient may not sublicense,
//  assign, transfer or otherwise provide the source code to any third
//  party. Qualcomm Technologies, Inc. retains all ownership rights in and
//  to the software
//
//  This notice supersedes any other QTI notices contained within the software
//  except copyright notices indicating different years of publication for
//  different portions of the software. This notice does not supersede the
//  application of any third party copyright notice to that third party's
//  code.
//
//  @@
//
//==============================================================================
//
// This file contains an example application that loads and executes a neural
// network using the SNPE C++ API and saves the layer output to a file.
// Inputs to and outputs from the network are conveyed in binary form as single
// precision floating point values.
//



#include <chrono>
#include <iostream>
#include <getopt.h>
#include "CheckRuntime.hpp"
#include "LoadContainer.hpp"
#include "SetBuilderOptions.hpp"
#include "LoadInputTensor.hpp"
#include "ExecuteNetwork.hpp"
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iterator>
#include "udlExample.hpp"
// Command line settings
int main(int argc, char** argv)
{
    //Process command line args
    static std::string dlc="";
    static std::string OutputDir="./output/";
    const char* inputFile="";
    int opt = 0;
    while ((opt = getopt(argc, argv, "hi:d:o:")) != -1)
    {
        switch (opt)
        {
            case 'h':
                std::cout
                    << "\nDESCRIPTION:\n"
                    << "------------\n"
                    << "Example application demonstrating how to load and execute a neural network\n"
                    << "using the SNPE C++ API.\n"
                    << "\n\n"
                    << "REQUIRED ARGUMENTS:\n"
                    << "-------------------\n"
                    << "  -d  <FILE>   Path to the DL container containing the network.\n"
                    << "  -i  <FILE>   Path to a file listing the inputs for the network.\n"
                    << "  -o  <PATH>   Path to directory to store output results.\n";

                std::exit(0);
            case 'i':
                inputFile = optarg;
                break;
            case 'd':
                dlc = optarg;
                break;
            case 'o':
                OutputDir = optarg;
                break;
            default:
                std::cout << "Invalid parameter specified. Please run snpe-sample with the -h flag to see required arguments" << std::endl;
                std::exit(0);
        }
    }

    //Check if given arguments represent valid files
    std::ifstream dlcFile(dlc);
    std::ifstream inputList(inputFile);
    if (!dlcFile || !inputList) {
        std::cout << "Input list or dlc file not valid. Please ensure that you have provided a valid input list and dlc for processing. Run snpe-sample with the -h flag for more details" << std::endl;
        std::exit(0);	
    }


    // Open the DL container that contains the network to execute.
    // Create an instance of the SNPE network from the now opened container.
    // The factory functions provided by SNPE allow for the specification
    // of which layers of the network should be returned as output and also
    // if the network should be run on the CPU or GPU.
    // The runtime availability API allows for runtime support to be queried.
    // If a selected runtime is not available, we will issue a warning and continue,
    // expecting the invalid configuration to be caught at SNPE network creation.

    zdl::DlSystem::UDLFactoryFunc udlFunc = sample::MyUDLFactory;
    // 0xdeadbeaf to test cookie
    zdl::DlSystem::UDLBundle udlBundle; udlBundle.cookie = (void*)0xdeadbeaf, udlBundle.func = udlFunc;

    static zdl::DlSystem::Runtime_t runtime = checkRuntime();
    std::cout << "runtime is : " << (int)runtime << std::endl;

    std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(dlc);
    std::unique_ptr<zdl::SNPE::SNPE> snpe = setBuilderOptions(container, runtime, udlBundle);

    // Open the input file listing and for each input file load its contents
    // into a SNPE tensor for a single input network, execute the network
    // with the input and save each of the returned output tensors to a file.
    size_t inputListNum = 0;
    std::string fileLine;

    while (std::getline(inputList, fileLine))
    {
        if (fileLine.empty()) continue;
        std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(snpe, fileLine);
        executeNetwork (snpe , inputTensor, OutputDir, inputListNum);
        executeNetwork (snpe , inputTensor, OutputDir, inputListNum);
        executeNetwork (snpe , inputTensor, OutputDir, inputListNum);

        auto  start = std::chrono::high_resolution_clock::now();
        executeNetwork (snpe , inputTensor, OutputDir, inputListNum);
        double time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "duration : " << time << std::endl;
        ++inputListNum;
    }

    return 0;
}
