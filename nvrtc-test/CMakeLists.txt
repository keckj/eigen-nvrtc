cmake_minimum_required(VERSION 3.18)
project(eigen_nvrtc_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

get_filename_component(EIGEN_NVRTC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_INC)

add_executable(eigen_nvrtc_test main.cpp)
target_include_directories(eigen_nvrtc_test PRIVATE ${EIGEN_NVRTC_ROOT})
target_compile_definitions(eigen_nvrtc_test PRIVATE
  EIGEN_NVRTC_ROOT="${EIGEN_NVRTC_ROOT}"
  CUDA_INCLUDE_DIR="${CUDA_INC}"
  TESTS_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/tests.cuh")
target_link_libraries(eigen_nvrtc_test PRIVATE CUDA::nvrtc CUDA::cuda_driver)

enable_testing()
add_test(NAME eigen_nvrtc_test COMMAND eigen_nvrtc_test)
