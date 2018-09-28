#ifndef SETBUILDEROPTIONS_H
#define SETBUILDEROPTIONS_H
#include "SNPE/SNPE.hpp"

#include "SNPE/SNPEFactory.hpp"

#include "DlContainer/IDlContainer.hpp"
#include "udlExample.hpp"




std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container , zdl::DlSystem::Runtime_t runtime, zdl::DlSystem::UDLBundle udlBundle);
#endif
