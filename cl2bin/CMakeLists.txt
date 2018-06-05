cmake_minimum_required(VERSION 2.8)
project (cl2bin)

set(CMAKE_MODULE_PATH /home/pengcuo/cmake)
include(toolchain)

add_executable(cl2bin ${CMAKE_CURRENT_SOURCE_DIR}/cl2bin.cpp)
target_link_libraries(cl2bin OpenCL)
