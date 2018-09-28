#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <string>
#include <cstdlib>
#include <sys/stat.h>
#include <cerrno>
#include "DlSystem/ITensorFactory.hpp"

bool EnsureDirectory(const std::string& dir)
{
   auto i = dir.find_last_of('/');
   std::string prefix = dir.substr(0, i);

   if (dir.empty() || dir == "." || dir == "..")
   {
      return true;
   }

   if (i != std::string::npos && !EnsureDirectory(prefix))
   {
      return false;
   }

   int rc = mkdir(dir.c_str(),  S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
   if (rc == -1 && errno != EEXIST)
   {
      return false;
   }
   else
   {
      struct stat st;
      if (stat(dir.c_str(), &st) == -1)
      {
         return false;
      }

      return S_ISDIR(st.st_mode);
   }
}

// Helper for splitting tokenized strings
std::vector<float> loadFloatDataFile(const std::string& inputFile)
{
    std::ifstream in(inputFile, std::ifstream::binary);
    if (!in.is_open() || !in.good())
    {
    std::cerr << "Failed to open input file: " << inputFile << "\n";
    }

    in.seekg(0, in.end);
    int length = in.tellg();
    in.seekg(0, in.beg);

    std::vector<float> vec;
    vec.resize(length/sizeof(float));
    if (!in.read(reinterpret_cast<char*>(&vec[0]), length))
    {
    std::cerr << "Failed to read the contents of: " << inputFile << "\n";
    }

    return vec;
    }

std::vector<unsigned char> loadByteDataFile(const std::string& inputFile)
{
   std::ifstream in(inputFile, std::ifstream::binary);
   if (!in.is_open() || !in.good())
   {
      std::cerr << "Failed to open input file: " << inputFile << "\n";
   }

   in.seekg(0, in.end);
   int length = in.tellg();
   in.seekg(0, in.beg);

   std::vector<unsigned char> vec;
   vec.resize(length);
   if (!in.read(reinterpret_cast<char*>(&vec[0]), length))
   {
      std::cerr << "Failed to read the contents of nv21 file: " << inputFile << "\n";
   }

   return vec;
}

void SaveITensor(const std::string& path, const zdl::DlSystem::ITensor* tensor)
{
   // Create the directory path if it does not exist
   auto idx = path.find_last_of('/');
   if (idx != std::string::npos)
   {
      std::string dir = path.substr(0, idx);
      if (!EnsureDirectory(dir))
      {
          std::cerr << "Failed to create output directory: " << dir << ": "
              << std::strerror(errno) << "\n";
          std::exit(EXIT_FAILURE);
      }
   }

   std::ofstream os(path, std::ofstream::binary);
   if (!os)
   {
      std::cerr << "Failed to open output file for writing: " << path << "\n";
      std::exit(EXIT_FAILURE);
   }

   for ( auto it = tensor->cbegin(); it != tensor->cend(); ++it )
   {
      float f = *it;
      if (!os.write(reinterpret_cast<char*>(&f), sizeof(float)))
      {
         std::cerr << "Failed to write data to: " << path << "\n";
         std::exit(EXIT_FAILURE);
      }
   }
}
