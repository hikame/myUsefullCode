cmake_minimum_required(VERSION 3.10)
project(DEMO)

option(TEST "compute TEST" ON)
option(DEBUG "compute DEBUG" OFF)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -s")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -s")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE : ${CMAKE_CXX_FLAGS_RELEASE}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE SRC "src/*cpp")

if(DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -Wall -Wno-unused-command-line-argument")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")


add_library(compute SHARED ${SRC})
add_library(compute_static STATIC ${SRC})
set_target_properties(compute_static PROPERTIES OUTPUT_NAME "compute")
set_target_properties(compute PROPERTIES CLEAN_DIRECT_OUTPUT 1)
set_target_properties(compute_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)

message(STATUS "DIR : ${CMAKE_CURRENT_BINARY_DIR}")
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)
install(TARGETS compute ARCHIVE DESTINATION lib)
install(TARGETS compute_static ARCHIVE DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/compute.hpp  DESTINATION include)

if(TEST)
    add_subdirectory(test)
endif()

