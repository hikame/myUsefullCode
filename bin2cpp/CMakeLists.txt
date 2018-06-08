cmake_minimum_required(VERSION 2.8)
project (bin2cpp)

SET(CMAKE_MODULE_PATH /home/pengcuo/cmake/)
include(toolchain)

add_executable(bin2cpp ${CMAKE_CURRENT_SOURCE_DIR}/bin2cpp.cpp)

file(GLOB_RECURSE src_list src/*cpp)
message(STATUS "src -> ${src_list}")
#add_library(oclKernel STATIC ${src_list})
#add_library(oclKernel SHARED ${src_list})
