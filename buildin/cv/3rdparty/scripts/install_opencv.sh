#!/bin/bash
set -e

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -d "opencv" ];then
  git clone https://github.com/opencv/opencv.git
fi

cd opencv

git checkout 4.5.1

mkdir -p build

cd build

rm -rf *

cmake .. \
    -DCMAKE_INSTALL_PREFIX=${CURRENT_DIR}/../edge/opencv \
    -DBUILD_LIST=core,imgcodecs,imgproc,videoio \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_TESTS=OFF \
    -DWITH_OPENMP=OFF \
    -DBUILD_ZLIB=OFF \
    -DWITH_PNG=ON \
    -DWITH_DC1394=OFF \
    -DWITH_FFMPEG=ON \
    -DBUILD_TESTS=OFF \
    -DARM=1 \
    -DENABLE_NEON=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CONFIGURATION_TYPES=Release \
    -DCMAKE_TOOLCHAIN_FILE=${CURRENT_DIR}/toolchain.cmake \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DOPENCV_FFMPEG_USE_FIND_PACKAGE=ON \
    -DFFMPEG_DIR=${CURRENT_DIR}/../edge/ffmpeg \
    -DOPENCV_VIDEOIO_DEBUG=1

make -j
make install
