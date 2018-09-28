#ifndef LOADCONTAINER_H
#define LOADCONTAINER_H
#include "DlContainer/IDlContainer.hpp"
#include <iostream>
std::unique_ptr<zdl::DlContainer::IDlContainer> loadContainerFromFile(std::string containerPath);
#endif
