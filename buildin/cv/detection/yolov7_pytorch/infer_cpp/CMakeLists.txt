cmake_minimum_required(VERSION 2.8)

project(yolov7_pytorch)

# message("-- ABI_MODE=$ENV{ABI_MODE}")
message("-- NEUWARE_HOME=$ENV{NEUWARE_HOME}")
message("-- TOOLCHAIN_DIR=$ENV{TOOLCHAIN_DIR}")

set(CMAKE_SYSTEM_NAME Linux)
set(DEP_DIR ${CMAKE_SOURCE_DIR}/../../../3rdparty)
# set(ABI_MODE $ENV{ABI_MODE})
set(CMAKE_CXX_FLAGS "-Werror -g -fPIC -std=c++11 -pthread -lglog -lgflags")
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${ABI_MODE})

include_directories(include $ENV{NEUWARE_HOME}/include ${DEP_DIR}/edge/CLI11)
link_directories($ENV{NEUWARE_HOME}/lib64)

if(DEFINED ENV{TOOLCHAIN_DIR})
    SET(EDGE_TOOL $ENV{TOOLCHAIN_DIR})
    SET(CMAKE_C_COMPILER "${EDGE_TOOL}/bin/aarch64-linux-gnu-gcc")
    SET(CMAKE_CXX_COMPILER "${EDGE_TOOL}/bin/aarch64-linux-gnu-g++")
    SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin/)
    include_directories(${DEP_DIR}/edge/)
    include_directories(${DEP_DIR}/edge/opencv/include/opencv4)
    include_directories(${DEP_DIR}/edge/opencv/include)
    include_directories(${DEP_DIR}/edge/glog/include)
    include_directories(${DEP_DIR}/edge/gflags/include)
    link_directories(${DEP_DIR}/edge/opencv/lib)
    link_directories(${DEP_DIR}/edge/glog/lib)
    link_directories(${DEP_DIR}/edge/gflags/lib)
    add_executable(edge_infer src/infer.cpp src/pre_process.cpp src/post_process.cpp src/utils.cpp src/coco_result_saver.cpp)
    target_link_libraries(edge_infer magicmind_runtime cnrt opencv_core opencv_imgcodecs opencv_imgproc)
endif()


