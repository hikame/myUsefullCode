cmake_minimum_required(VERSION 2.8)
project (bin2cpp)

SET(CMAKE_MODULE_PATH /home/pengcuo/cmake/)
include(toolchain)

add_executable(bin2cpp ${CMAKE_CURRENT_SOURCE_DIR}/bin2cpp.cpp)
