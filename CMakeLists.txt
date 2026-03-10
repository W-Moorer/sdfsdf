cmake_minimum_required(VERSION 3.16)
project(VolumetricDynamicsEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
include(FetchContent)

# Third-party dependencies should not pollute the project's root CTest entry.
set(VDE_BUILD_TESTING ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "Disable third-party tests during dependency configuration" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen documentation build" FORCE)
set(GLM_BUILD_TESTS OFF CACHE BOOL "Disable GLM tests" FORCE)

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

set(BUILD_TESTING ${VDE_BUILD_TESTING} CACHE BOOL "Enable VolumetricDynamicsEngine tests" FORCE)

include_directories(${eigen_SOURCE_DIR})

add_library(vde_core INTERFACE)
target_include_directories(vde_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(vde_core INTERFACE
    polyscope
)

add_subdirectory(src)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
add_subdirectory(benchmarks)
add_subdirectory(examples)
