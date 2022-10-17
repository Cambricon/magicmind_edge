# 编译c++代码，在当前目录输出x86可执行文件video_action_recognize_x86.bin
set -e 

export NEUWARE_HOME=/usr/local/neuware/edge
export TOOLCHAIN_DIR=/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu

mkdir -p build
rm -rf build/*
cd build
cmake ..
make
cd ..

