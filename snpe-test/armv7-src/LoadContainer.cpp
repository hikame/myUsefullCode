#include <iostream>
#include "SNPE/SNPE.hpp"

#include "SNPE/SNPEFactory.hpp"

#include "DlContainer/IDlContainer.hpp"


// Command line settings
std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromFile(std::string containerPath)
{
    std::unique_ptr<zdl::DlContainer::IDlContainer> container;
    container = zdl::DlContainer::IDlContainer::open(containerPath);
    return container;
}
