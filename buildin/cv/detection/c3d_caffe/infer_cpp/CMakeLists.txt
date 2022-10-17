cmake_minimum_required(VERSION 2.8)

project(yolov3_caffe)

message("-- NEUWARE_HOME=$ENV{NEUWARE_HOME}")
message("-- TOOLCHAIN_DIR=$ENV{TOOLCHAIN_DIR}")

set(CMAKE_SYSTEM_NAME Linux)
set(DEP_DIR ${CMAKE_SOURCE_DIR}/../../../3rdparty)
set(NEUWARE_HOME $ENV{NEUWARE_HOME})
set(CMAKE_CXX_FLAGS "-Werror -g -fPIC -std=c++11")

SET(EDGE_TOOL $ENV{TOOLCHAIN_DIR})
SET(CMAKE_C_COMPILER "${EDGE_TOOL}/bin/aarch64-linux-gnu-gcc")
SET(CMAKE_CXX_COMPILER "${EDGE_TOOL}/bin/aarch64-linux-gnu-g++")
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin/)

include_directories(include
    ${NEUWARE_HOME}/include
    ${DEP_DIR}/edge/
    ${DEP_DIR}/edge/CLI11
    ${DEP_DIR}/edge/opencv/include/opencv4
    ${DEP_DIR}/edge/glog/include
    ${DEP_DIR}/edge/gflags/include
)

link_directories(
    ${NEUWARE_HOME}/lib64
    ${DEP_DIR}/edge/opencv/lib
    ${DEP_DIR}/edge/glog/lib
    ${DEP_DIR}/edge/gflags/lib
    ${DEP_DIR}/edge/ffmpeg/lib
)

add_executable(edge_infer src/infer.cpp src/pre_process.cpp src/post_process.cpp src/utils.cpp)
target_link_libraries(edge_infer
    magicmind_runtime
    cnrt
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    opencv_videoio
    pthread
    glog
    gflags
)
