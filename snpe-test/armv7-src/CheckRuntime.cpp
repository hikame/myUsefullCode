#include <iostream>
#include "SNPE/SNPE.hpp"

#include "SNPE/SNPEFactory.hpp"



// Command line settings
zdl::DlSystem::Runtime_t checkRuntime()
{
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    static zdl::DlSystem::Runtime_t Runtime;
    std::cout << "SNPE Version: " << Version.toString() << std::endl; //Print Version number
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU)) {
        Runtime = zdl::DlSystem::Runtime_t::GPU;
    }
    else{
        Runtime = zdl::DlSystem::Runtime_t::CPU;
    }
    return Runtime;
}
