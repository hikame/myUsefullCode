#ifndef UTIL_H
#define UTIL_H
#include <vector>
#include <string>
#include <sstream>

#include "SNPE/SNPE.hpp"
#include "DlSystem/ITensorFactory.hpp"
template <typename Container> Container& split(Container& result, const typename Container::value_type & s, typename Container::value_type::value_type delimiter )
{
  result.clear();
  std::istringstream ss( s );
  while (!ss.eof())
  {
    typename Container::value_type field;
    getline( ss, field, delimiter );
    if (field.empty()) continue;
    result.push_back( field );
  }
  return result;
}


std::vector<float> loadFloatDataFile(const std::string &inputFile);

std::vector<unsigned char> loadByteDataFile(const std::string& inputFile);
void SaveITensor(const std::string& path, const zdl::DlSystem::ITensor* tensor);
bool EnsureDirectory(const std::string& dir);

#endif

