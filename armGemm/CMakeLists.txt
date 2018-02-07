cmake_minimum_required(VERSION 2.8)
include(hpcc-common)
message(STATUS "CMAKE_HOME_DIRECTORY--->${CMAKE_HOME_DIRECTORY}")
include_directories(${CMAKE_HOME_DIRECTORY}/include)
link_directories(${CMAKE_HOME_DIRECTORY}/lib)

add_executable(gemm_speed gemm_neon.cpp)
target_link_libraries(gemm_speed fmath_android_armv7)
