cmake_minimum_required(VERSION 2.8)

project (clInfo)
set(CMAKE_MODULE_PATH /home/pengcuo/cmake)
include(toolchain)

message(STATUS "cxx_compile_flags---> ${CMAKE_CXX_FLAGS}")
add_executable(clInfo ${CMAKE_CURRENT_SOURCE_DIR}/src/clInfo.cpp)
target_link_libraries(clInfo OpenCL)
