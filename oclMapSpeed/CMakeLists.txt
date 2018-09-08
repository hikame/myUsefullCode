cmake_minimum_required(VERSION 2.8)

set(TOOLDIR ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/)
set(CMAKE_MODULE_PATH ${TOOLDIR})
include(toolchain)

project(oclMapSpeed)
add_executable(oclMapSpeed oclMapSpeed.cpp)
target_link_libraries(oclMapSpeed OpenCL)
