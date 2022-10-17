cmake_minimum_required(VERSION 3.5)

project(edge_modelzoo_server)

message("-- ABI_MODE=$ENV{ABI_MODE}")
message("-- NEUWARE_HOME=$ENV{NEUWARE_HOME}")
message("-- TOOLCHAIN_DIR=$ENV{TOOLCHAIN_DIR}")

set(CMAKE_SYSTEM_NAME Linux)
set(ABI_MODE $ENV{ABI_MODE})
set(EDGE_TOOL $ENV{TOOLCHAIN_DIR})
set(CMAKE_C_COMPILER "${EDGE_TOOL}/bin/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${EDGE_TOOL}/bin/aarch64-linux-gnu-g++")
set(CMAKE_CXX_FLAGS "-Werror -g -fPIC -std=c++11 -pthread")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--enable-new-dtags")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${ABI_MODE})

include_directories($ENV{NEUWARE_HOME}/include)
link_directories($ENV{NEUWARE_HOME}/lib64)

add_executable(edge_server ../sample_server.cc)
target_link_libraries(edge_server magicmind_runtime)
