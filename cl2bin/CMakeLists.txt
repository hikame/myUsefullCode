cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH /home/pengcuo/cmake)
include(toolchain)

project (cl2bin)
add_executable(cl2bin ${CMAKE_CURRENT_SOURCE_DIR}/cl2bin.cpp)
target_link_libraries(cl2bin OpenCL)
