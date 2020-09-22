set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "")
include_directories(${PROJECT_SOURCE_DIR}/include)

set(LINK_LIBS "log z jnigraphics android jnigraphics mediandk")

add_executable(test_demo test.cpp)
target_link_libraries(test_demo compute_static log z jnigraphics android jnigraphics mediandk) 

