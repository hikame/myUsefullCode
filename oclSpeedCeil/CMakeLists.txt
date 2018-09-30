cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH /home/pengcuo/cmake)
include(toolchain)

project(opencl_ceil)
add_executable(opencl_ceil ceil.cpp)
target_link_libraries(opencl_ceil OpenCL)
