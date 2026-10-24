cmake_minimum_required(VERSION 3.16)
project(qwen_slam CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
add_definitions(-DCOMPILEDWITHC11)
set(ORB $ENV{HOME}/slam/ORB_SLAM3)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Pangolin REQUIRED)
add_executable(mono_live mono_live.cc)
target_include_directories(mono_live PRIVATE ${ORB} ${ORB}/include ${ORB}/include/CameraModels
                           ${ORB}/Thirdparty/Sophus ${Pangolin_INCLUDE_DIRS})
target_link_libraries(mono_live ${ORB}/lib/libORB_SLAM3.so ${OpenCV_LIBS} Eigen3::Eigen
                      ${Pangolin_LIBRARIES} -lboost_serialization -lcrypto)
add_executable(map_offline map_offline.cc)
target_include_directories(map_offline PRIVATE ${ORB} ${ORB}/include ${ORB}/include/CameraModels
                           ${ORB}/Thirdparty/Sophus ${Pangolin_INCLUDE_DIRS})
target_link_libraries(map_offline ${ORB}/lib/libORB_SLAM3.so ${OpenCV_LIBS} Eigen3::Eigen
                      ${Pangolin_LIBRARIES} -lboost_serialization -lcrypto)
