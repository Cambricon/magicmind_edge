# !/bin/bash

export PATH=/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/bin:$PATH

rm -r opencv
mkdir -p ./opencv
# opencv
wget -c https://github.com/opencv/opencv/archive/3.2.0.zip
if [ $? -ne 0 ]; then
    echo "download opencv failed."
    exit 1
fi
unzip 3.2.0.zip
rm 3.2.0.zip
cd opencv-3.2.0
mkdir -p build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../../toolchain-edge.cmake \
         -DCMAKE_INSTALL_PREFIX=../../opencv  \
         -DWITH_CUDA=OFF -DWITH_OPENCL=OFF -DWITH_GPHOTO2=OFF \
         -DBUILD_DOCS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF \
         -DBUILD_WITH_DEBUG_INFO=OFF -DBUILD_opencv_apps=off -DWITH_V4L=OFF -DWITH_GTK=OFF \
         -DCMAKE_VERBOSE=on -DWITH_LIBV4L=OFF -DWITH_1394=OFF -DWITH_TIFF=OFF -DWITH_OPENEXR=OFF \
         -DBUILD_OPENEXR=OFF -DBUILD_opencv_ocl=off -DWITH_GSTREAMER=OFF -DWITH_FFMPEG=OFF \
         -DWITH_EIGEN=OFF -DWITH_GIGEAPI=OFF -DWITH_JASPER=OFF -DWITH_CUFFT=OFF \
         -DBUILD_opencv_contrib=off -DBUILD_opencv_ml=OFF -DBUILD_opencv_objdetect=OFF \
         -DBUILD_opencv_nonfree=off -DBUILD_opencv_gpu=OFF -DWITH_PNG=OFF 
if [ $? -ne 0 ]; then
    echo "compile opencv failed."
    exit 1
fi
make -j4
if [ $? -ne 0 ]; then
    echo "compile opencv failed."
    exit 1
fi
make install
if [ $? -ne 0 ]; then
    echo "install opencv failed."
    exit 1
fi
cd ../../
rm -r opencv-3.2.0
rm -r opencv/share

# gflags
wget -c https://github.com/gflags/gflags/archive/refs/tags/v2.2.1.zip
if [ $? -ne 0 ]; then
    echo "download gflags failed."
    exit 1
fi
unzip v2.2.1.zip
rm v2.2.1.zip

rm -r gflags
mkdir gflags
cd gflags-2.2.1
mkdir -p build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../../toolchain-edge.cmake \
         -DCMAKE_INSTALL_PREFIX=../../gflags/ \
         -DBUILD_SHARED_LIBS=on \
         -DBUILD_STATIC_LIBS=off \
         -DBUILD_gflags_LIB=on \
         -DINSTALL_STATIC_LIBS=off \
         -DINSTALL_SHARED_LIBS=on \
         -DREGISTER_INSTALL_PREFIX=off
if [ $? -ne 0 ]; then
    echo "compile gflags failed."
    exit 1
fi
make -j4
if [ $? -ne 0 ]; then
    echo "compile gflags failed."
    exit 1
fi
make install
if [ $? -ne 0 ]; then
    echo "install gflags failed."
    exit 1
fi
cd ../../
rm -r gflags-2.2.1
rm -r gflags/bin/

## glog
wget -c https://github.com/google/glog/archive/refs/tags/v0.5.0.zip
if [ $? -ne 0 ]; then
    echo "download glog failed."
    exit 1
fi
unzip v0.5.0.zip
rm v0.5.0.zip
rm -r glog
mkdir glog
cd glog-0.5.0
mkdir -p build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../../toolchain-edge.cmake \
         -DCMAKE_FIND_ROOT_PATH=../../glog  \
         -DCMAKE_INSTALL_PREFIX=../../glog \
         -DBUILD_SHARED_LIBS=on
if [ $? -ne 0 ]; then
    echo "compile glog failed."
    exit 1
fi
make -j4
if [ $? -ne 0 ]; then
    echo "compile glog failed."
    exit 1
fi
make install
if [ $? -ne 0 ]; then
    echo "install glog failed."
    exit 1
fi
cd ../../
rm -r glog-0.5.0
echo "Build and install 3rdparty success."

