#!/bin/bash
# Build Pangolin + ORB-SLAM3 (library only) on the Pi 4, Debian Trixie / GCC 14.
# Log: ~/slam/build.log
set -euxo pipefail
mkdir -p ~/slam && cd ~/slam

sudo apt-get update
sudo apt-get install -y --no-install-recommends git cmake build-essential pkg-config \
    libeigen3-dev libopencv-dev libboost-serialization-dev libssl-dev \
    libglew-dev libepoxy-dev libgl1-mesa-dev libegl1-mesa-dev \
    libwayland-dev libxkbcommon-dev wayland-protocols

# Pangolin (ORB-SLAM3's viewer library; we run headless but the code links it)
if [ ! -f /usr/local/lib/libpango_core.so ]; then
    [ -d Pangolin ] || git clone --depth 1 --branch v0.9.1 https://github.com/stevenlovegrove/Pangolin.git
    cmake -S Pangolin -B Pangolin/build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF \
        -DBUILD_TESTS=OFF -DBUILD_TOOLS=OFF -DBUILD_PANGOLIN_PYTHON=OFF \
        -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_LIBUVC=OFF -DBUILD_PANGOLIN_REALSENSE2=OFF
    cmake --build Pangolin/build -j3
    sudo cmake --install Pangolin/build
    sudo ldconfig
fi

[ -d ORB_SLAM3 ] || git clone --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3

# GCC 14 / C++17 fixes (idempotent)
sed -i 's/std::pair<const KeyFrame\*, g2o::Sim3>/std::pair<KeyFrame* const, g2o::Sim3>/' include/LoopClosing.h
sed -i "s/bool mnFullBAIdx;/int mnFullBAIdx;/" include/LoopClosing.h   # ++ on bool is illegal in C++17
for f in CMakeLists.txt Thirdparty/DBoW2/CMakeLists.txt Thirdparty/g2o/CMakeLists.txt; do
    sed -i 's/-march=native/-mcpu=cortex-a72/g' "$f"
done
# Force C++17 and keep the COMPILEDWITHC11 code paths (steady_clock instead of monotonic_clock)
grep -q 'CMAKE_CXX_STANDARD 17' CMakeLists.txt || sed -i '/^project(ORB_SLAM3)/a set(CMAKE_CXX_STANDARD 17)\nadd_definitions(-DCOMPILEDWITHC11)' CMakeLists.txt
sed -i 's/CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)/set(COMPILER_SUPPORTS_CXX11 TRUE)/' CMakeLists.txt
sed -i 's/-std=c++11/-std=c++17/g; s/-std=c++0x/-std=c++17/g' CMakeLists.txt
sed -i 's/find_package(OpenCV 4.4)/find_package(OpenCV 4)/' CMakeLists.txt

for lib in DBoW2 g2o; do
    cmake -S Thirdparty/$lib -B Thirdparty/$lib/build -DCMAKE_BUILD_TYPE=Release
    cmake --build Thirdparty/$lib/build -j3
done
cmake -S Thirdparty/Sophus -B Thirdparty/Sophus/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
cmake --build Thirdparty/Sophus/build -j3

[ -f Vocabulary/ORBvoc.txt ] || (cd Vocabulary && tar -xf ORBvoc.txt.tar.gz)

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# Library + the monocular example only (the full example set adds an hour)
cmake --build build -j1 --target ORB_SLAM3 mono_tum -- -k   # -k: report every error in one pass
ls -la lib/
echo BUILD_OK
