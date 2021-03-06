//==============================================================================
//
//  Copyright (c) 2015-2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
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
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iterator>
#include <unordered_map>

#include "CheckRuntime.hpp"
#include "LoadContainer.hpp"
#include "SetBuilderOptions.hpp"
#include "LoadInputTensor.hpp"
#include "udlExample.hpp"
#include "CreateUserBuffer.hpp"
#include "PreprocessInput.hpp"
#include "SaveOutputTensor.hpp"
#ifdef ANDROID
#include <GLES2/gl2.h>
#include "CreateGLBuffer.hpp"
#endif

#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/UDLFunc.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"
#include "DiagLog/IDiagLog.hpp"

int main(int argc, char** argv)
{
    enum {UNKNOWN, USERBUFFER, ITENSOR};
    enum {CPUBUFFER, GLBUFFER};

    // Command line arguments
    static std::string dlc = "";
    static std::string OutputDir = "./output/";
    const char* inputFile = "";
    std::string bufferTypeStr = "USERBUFFER";
    std::string userBufferSourceStr = "CPUBUFFER";

    // Process command line arguments
    int opt = 0;
    while ((opt = getopt(argc, argv, "hi:d:o:b:s:")) != -1)
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
                        << "  -o  <PATH>   Path to directory to store output results.\n"
                        << "\n"
                        << "OPTIONAL ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  -b  <TYPE>   Type of buffers to use [USERBUFFER, ITENSOR] (" << bufferTypeStr << " is default).\n"
#ifdef ANDROID
                        << "  -s  <TYPE>   Source of user buffers to use [GLBUFFER, CPUBUFFER] (" << userBufferSourceStr << " is default).\n"
                        << "               GL buffer is only supported on Android OS.\n"
#endif
                        << std::endl;

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
            case 'b':
                bufferTypeStr = optarg;
                break;
            case 's':
                userBufferSourceStr = optarg;
                break;
            default:
                std::cout << "Invalid parameter specified. Please run snpe-sample with the -h flag to see required arguments" << std::endl;
                std::exit(0);
        }
    }

    // Check if given arguments represent valid files
    std::ifstream dlcFile(dlc);
    std::ifstream inputList(inputFile);
    if (!dlcFile || !inputList) {
        std::cout << "Input list or dlc file not valid. Please ensure that you have provided a valid input list and dlc for processing. Run snpe-sample with the -h flag for more details" << std::endl;
        std::exit(0);
    }

    // Check if given buffer type is valid
    int bufferType;
    if (bufferTypeStr == "USERBUFFER") {
        bufferType = USERBUFFER;
    } else if (bufferTypeStr == "ITENSOR") {
        bufferType = ITENSOR;
    } else {
        std::cout << "Buffer type is not valid. Please run snpe-sample with the -h flag for more details" << std::endl;
        std::exit(0);
    }
    //Check if given user buffer source type is valid
    int userBufferSourceType;
    if (userBufferSourceStr == "CPUBUFFER")
    {
        userBufferSourceType = CPUBUFFER;
    } else if (userBufferSourceStr == "GLBUFFER") {
#ifndef ANDROID
        std::cout << "GLBUFFER mode is only supported on Android OS" << std::endl;
        std::exit(0);
#endif
        userBufferSourceType = GLBUFFER;
    } else {
        std::cout << "Source of user buffer type is not valid. Please run snpe-sample with the -h flag for more details" << std::endl;
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
    zdl::DlSystem::UDLBundle udlBundle; udlBundle.cookie = (void*)0xdeadbeaf, udlBundle.func = udlFunc; // 0xdeadbeaf to test cookie

    static zdl::DlSystem::Runtime_t runtime = checkRuntime();
    int run = (int)runtime;
    std::cout << "runtime : " << run << std::endl;
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(dlc);
    bool useUserSuppliedBuffers = (bufferType == USERBUFFER);

    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::DlSystem::PlatformConfig platformConfig;
#ifdef ANDROID
    CreateGLBuffer* glBuffer = new CreateGLBuffer();
    if (userBufferSourceType == GLBUFFER)
    {
      glBuffer->SetGPUPlatformConfig(platformConfig);
    }
#endif
    snpe = setBuilderOptions(container, runtime, udlBundle, useUserSuppliedBuffers, platformConfig);

    // Configure logging output and start logging. The snpe-diagview
    // executable can be used to read the content of this diagnostics file
    auto logger_opt = snpe->getDiagLogInterface();
    if (!logger_opt) throw std::runtime_error("SNPE failed to obtain logging interface");
    auto logger = *logger_opt;
    auto opts = logger->getOptions();

    opts.LogFileDirectory = OutputDir;
    if(!logger->setOptions(opts)) {
        std::cerr << "Failed to set options" << std::endl;
        std::exit(1);
    }
    if (!logger->start()) {
        std::cerr << "Failed to start logger" << std::endl;
        std::exit(1);
    }

    // Check the batch size for the container
    // SNPE 1.16.0 (and newer) assumes the first dimension of the tensor shape
    // is the batch size.
    zdl::DlSystem::TensorShape tensorShape;
    tensorShape = snpe->getInputDimensions();
    size_t batchSize = tensorShape.getDimensions()[0];
#ifdef ANDROID
    if (userBufferSourceType == GLBUFFER)
    {
        if(tensorShape.rank() > 3) {
            std::cerr << "GL buffer source mode does not support batchsize larger than 1" << std::endl;
            std::exit(1);
        }
        else {
            batchSize = 1;
        }
    }
#endif
    std::cout << "Batch size for the container is " << batchSize << std::endl;

    // Open the input file listing and group input files into batches
    std::vector<std::vector<std::string>> inputs = preprocessInput(inputFile, batchSize);

    // Load contents of input file batches ino a SNPE tensor or user buffer,
    // execute the network with the input and save each of the returned output to a file.
    if(bufferType == USERBUFFER) {
        // SNPE allows its input and output buffers that are fed to the network
        // to come from user-backed buffers. First, SNPE buffers are created from
        // user-backed storage. These SNPE buffers are then supplied to the network
        // and the results are stored in user-backed output buffers. This allows for
        // reusing the same buffers for multiple inputs and outputs.
        zdl::DlSystem::UserBufferMap inputMap, outputMap;
        std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
        std::unordered_map<std::string, std::vector<uint8_t>> applicationOutputBuffers;
        createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, snpe);

                std::cout << __FILE__ << " " << __LINE__ << std::endl;
        if (userBufferSourceType == CPUBUFFER)
        {
            std::unordered_map<std::string, std::vector<uint8_t>> applicationInputBuffers;
            createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe);

            for (size_t i = 0; i < inputs.size(); i++) {
                // Load input user buffer(s) with values from file(s)
                if(batchSize > 1)
                    std::cout << "Batch " << i << ":" << std::endl;
                loadInputUserBuffer(applicationInputBuffers, snpe, inputs[i]);
                // Execute the input buffer map on the model with SNPE
                snpe->execute(inputMap, outputMap);
                snpe->execute(inputMap, outputMap);
                snpe->execute(inputMap, outputMap);
                snpe->execute(inputMap, outputMap);
                snpe->execute(inputMap, outputMap);

                std::cout << __FILE__ << " " << __LINE__ << std::endl;
                auto start = std::chrono::high_resolution_clock::now();
                snpe->execute(inputMap, outputMap);
                double time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
                std::cout << "time " << time << std::endl;
                std::cout << __FILE__ << " " << __LINE__ << std::endl;
                // Save the execution results
                saveOutput(outputMap, applicationOutputBuffers, OutputDir, i*batchSize, batchSize);

            }
        }
#ifdef ANDROID
        if(userBufferSourceType  == GLBUFFER)
        {
                std::cout << __FILE__ << " " << __LINE__ << std::endl;
            std::unordered_map<std::string, GLuint> applicationInputBuffers;
            createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe);
            GLuint glBuffers = 0;
            for(size_t i = 0; i < inputs.size(); i++) {
                glBuffers = glBuffer->convertImage2GLBuffer(inputs[i]);
                loadInputUserBuffer(applicationInputBuffers, snpe, glBuffers);
                std::cout << __FILE__ << " " << __LINE__ << std::endl;

                snpe->execute(inputMap, outputMap);
                saveOutput(outputMap, applicationOutputBuffers, OutputDir, i*batchSize, batchSize);
                glDeleteBuffers(1, &glBuffers);
            }
        }
#endif
    } else if(bufferType == ITENSOR) {

        // A tensor map for SNPE execution outputs
        zdl::DlSystem::TensorMap outputTensorMap;

        for (size_t i = 0; i < inputs.size(); i++) {
            // Load input/output buffers with ITensor
            if(batchSize > 1)
                std::cout << "Batch " << i << ":" << std::endl;
            std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(snpe, inputs[i]);
            // Execute the input tensor on the model with SNPE
            snpe->execute(inputTensor.get(), outputTensorMap);
            // Save the execution results
            saveOutput(outputTensorMap,OutputDir,i*batchSize, batchSize);
        }
    }
    return 0;
}
