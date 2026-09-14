cmake_minimum_required(VERSION 3.17)
project(FLUX_GENERATOR LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
set(CMAKE_CXX_STANDARD "17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(CUDAToolkit REQUIRED)

set(FLUX_PROJECT_DIR ${PROJECT_SOURCE_DIR}/../../)
set(_FLUX_CCCL_HINTS "")
foreach(_cuda_inc IN LISTS CUDAToolkit_INCLUDE_DIRS)
  list(APPEND _FLUX_CCCL_HINTS "${_cuda_inc}/cccl")
endforeach()
if(DEFINED CUDAToolkit_TARGET_DIR)
  list(APPEND _FLUX_CCCL_HINTS "${CUDAToolkit_TARGET_DIR}/include/cccl")
endif()
if(DEFINED CMAKE_CXX_COMPILER)
  # The generator is a host-C++ project, so CMAKE_CUDA_COMPILER_TOOLKIT_ROOT
  # may not be available. CUDA_HOME is the most reliable extra hint here.
  if(DEFINED ENV{CUDA_HOME})
    list(APPEND _FLUX_CCCL_HINTS "$ENV{CUDA_HOME}/include/cccl")
  endif()
endif()
unset(FLUX_CCCL_INCLUDE CACHE)
find_path(FLUX_CCCL_INCLUDE
  NAMES cuda/std/utility
  HINTS ${_FLUX_CCCL_HINTS})
if(FLUX_CCCL_INCLUDE)
  message(STATUS "Generator: using CCCL include ${FLUX_CCCL_INCLUDE}")
else()
  set(FLUX_CCCL_INCLUDE "")
endif()

include_directories(
  ${CUDAToolkit_INCLUDE_DIRS}
  ${FLUX_CCCL_INCLUDE}
  ${FLUX_PROJECT_DIR}/include
  ${FLUX_PROJECT_DIR}/3rdparty/cutlass/include
  ${FLUX_PROJECT_DIR}/3rdparty/cutlass/tools/util/include
)

add_executable(gen_comm_none gen_comm_none.cc)
add_executable(gen_ag_gemm gen_ag_gemm.cc)
add_executable(gen_gemm_rs gen_gemm_rs.cc)
add_executable(gen_a2a_transpose_gemm gen_a2a_transpose_gemm.cc)
add_executable(gen_moe_ag_scatter gen_moe_ag_scatter.cc)
add_executable(gen_moe_gather_rs gen_moe_gather_rs.cc)
add_executable(gen_gemm_a2a_transpose gen_gemm_a2a_transpose.cc)
