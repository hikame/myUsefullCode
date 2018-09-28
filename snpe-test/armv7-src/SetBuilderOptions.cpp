#include <iostream>
#include "SNPE/SNPE.hpp"

#include "SNPE/SNPEFactory.hpp"

#include "DlContainer/IDlContainer.hpp"

#include "SNPE/SNPEBuilder.hpp"
std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container , zdl::DlSystem::Runtime_t runtime, zdl::DlSystem::UDLBundle udlBundle)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    snpe = snpeBuilder.setOutputLayers({})
       .setRuntimeProcessor(runtime)
       .setUdlBundle(udlBundle)
       .build();

    return snpe;
}
