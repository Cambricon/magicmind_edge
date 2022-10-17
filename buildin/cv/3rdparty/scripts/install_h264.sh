set -e

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd x264

./configure \
    --enable-shared \
    --enable-static \
    --disable-asm \
    --prefix=${CURRENT_DIR}/../edge/h264 \
    --host=arm-linux \
    --cross-prefix=/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu- \

make
make install