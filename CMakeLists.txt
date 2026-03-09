cmake_minimum_required(VERSION 3.16)
project(VolumetricDynamicsEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)

FetchContent_Declare(
    polyscope
    GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
    GIT_TAG v2.2.1
)

FetchContent_MakeAvailable(eigen polyscope)

include_directories(${eigen_SOURCE_DIR})

add_library(vde_core INTERFACE)
target_include_directories(vde_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(vde_core INTERFACE
    polyscope
)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)
