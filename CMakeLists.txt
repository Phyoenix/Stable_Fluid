cmake_minimum_required(VERSION 3.20)
project(hellocuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/OpenVDB")
find_package(OpenVDB REQUIRED)
message(STATUS "Found OpenVDB ${OpenVDB_VERSION} at ${OpenVDB_LIBRARIES}")

add_subdirectory(src)
