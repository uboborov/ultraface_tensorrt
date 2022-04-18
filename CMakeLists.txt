cmake_minimum_required(VERSION 2.6)

project(ultraface)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

enable_language(CUDA)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
link_directories(/usr/local/cuda/lib64)
# tensorrt
#include_directories(/usr/include/x86_64-linux-gnu/)
#link_directories(/usr/lib/x86_64-linux-gnu/)

add_library(preprocess SHARED ${PROJECT_SOURCE_DIR}/preprocess.cu)
target_link_libraries(preprocess ${CUDA_LIBS} ${OpenCV_LIBS} cudart)

add_executable(ultraface ${PROJECT_SOURCE_DIR}/ultraface.cpp)
target_link_libraries(ultraface ${CUDA_LIBS} ${OpenCV_LIBS} nvinfer nvparsers nvonnxparser preprocess cudart)

add_definitions(-O2)

