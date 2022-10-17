set -e

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ ! -d "FFMPEG" ];then
  git clone https://github.com/FFmpeg/FFmpeg.git
fi

cd FFMPEG

git checkout n4.2.3

./configure \
    --prefix=${CURRENT_DIR}/../edge/ffmpeg \
    --disable-doc \
    --arch=aarch64 \
    --enable-shared \
    --disable-static \
    --enable-avresample \
    --cross-prefix=/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu- \
    --target-os=linux

make clean

make -j

make install

cp ffmpeg-config.cmake ../edge/ffmpeg/