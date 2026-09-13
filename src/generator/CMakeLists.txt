cmake_minimum_required(VERSION 3.17)
project(FLUX_GENERATOR LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
set(CMAKE_CXX_STANDARD "17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(CUDAToolkit REQUIRED)

set(FLUX_PROJECT_DIR ${PROJECT_SOURCE_DIR}/../../)
set(FLUX_CCCL_INCLUDE "${CUDAToolkit_INCLUDE_DIRS}/cccl")
if(EXISTS "${FLUX_CCCL_INCLUDE}/cuda/std/utility")
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
