cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH /home/pengcuo/cmake)
include(toolchain)

add_executable(bufferVSImage bufferVSImage.cpp)
target_link_libraries(bufferVSImage OpenCL)
