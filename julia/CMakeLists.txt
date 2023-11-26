project(ghostbasil)

cmake_minimum_required(VERSION 3.5)
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

find_package(JlCxx REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${CMAKE_SOURCE_DIR}/../ghostbasil/include)

add_library(ghostbasil_wrap SHARED ghostbasil_wrap.cpp)
target_link_libraries(ghostbasil_wrap JlCxx::cxxwrap_julia Eigen3::Eigen)
install(TARGETS ghostbasil_wrap
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib)