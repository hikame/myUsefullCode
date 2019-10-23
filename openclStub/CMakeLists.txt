project(opencl_stub)

cmake_minimum_required(VERSION 3.0)

file(GLOB_RECURSE C_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
add_library(OpenCL SHARED ${C_SRCS})
target_include_directories(OpenCL PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
