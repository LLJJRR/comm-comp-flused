cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(FLUX LANGUAGES C CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules/")

# cmake global settings
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")
set(BUILD_THS
    ON
    CACHE INTERNAL "Build Torch op")
set(BUILD_TEST
    ON
    CACHE INTERNAL "Build unit tests")
set(ENABLE_NVSHMEM
    ON
    CACHE INTERNAL "Use NVSHMEM to transfer data")
set(CUTLASS_TRACE
    OFF
    CACHE INTERNAL "Print CUTLASS Host Trace info")
set(FLUX_DEBUG
    OFF
    CACHE INTERNAL "Define FLUX_DEBUG")
option(WITH_PROTOBUF "build with protobuf" OFF)
option(WITH_TRITON_AOT "build with triton Ahead-of-Time" OFF)
set(FLUX_BUILD_SCOPE "all" CACHE STRING
    "Build scope: all, comm_none, ag_gemm, gemm_rs, gemm_a2a_transpose, a2a_transpose_gemm, moe_ag_scatter, moe_gather_rs")
set_property(CACHE FLUX_BUILD_SCOPE PROPERTY STRINGS
    all comm_none ag_gemm gemm_rs gemm_a2a_transpose a2a_transpose_gemm
    moe_ag_scatter moe_gather_rs)
set(NCCL_ROOT
    "${PROJECT_SOURCE_DIR}/3rdparty/nccl/build/local"
    CACHE PATH "NCCL installation root")
option(ENABLE_GIN_AG "build NCCL GIN Rail-Ring AG+GEMM fusion" OFF)
set(NCCL_DEVICE_INCLUDE_DIR "$ENV{NCCL_DEVICE_INCLUDE_DIR}" CACHE PATH
    "NCCL 2.30.x source/internal include directory containing nccl_device.h")
set(NCCL_PUBLIC_INCLUDE_DIR "$ENV{NCCL_PUBLIC_INCLUDE_DIR}" CACHE PATH
    "NCCL installed include directory containing nccl.h")
if(ENABLE_GIN_AG)
  if(NOT NCCL_DEVICE_INCLUDE_DIR AND DEFINED ENV{NCCL_ROOT})
    if(EXISTS "$ENV{NCCL_ROOT}/src/include/nccl_device.h")
      set(NCCL_DEVICE_INCLUDE_DIR "$ENV{NCCL_ROOT}/src/include")
    elseif(EXISTS "$ENV{NCCL_ROOT}/include/nccl/detail/include/nccl_device.h")
      set(NCCL_DEVICE_INCLUDE_DIR "$ENV{NCCL_ROOT}/include/nccl/detail/include")
    endif()
  endif()
  if(NOT EXISTS "${NCCL_DEVICE_INCLUDE_DIR}/nccl_device.h")
    message(FATAL_ERROR
      "ENABLE_GIN_AG requires NCCL 2.30.x Device API headers. Set "
      "NCCL_DEVICE_INCLUDE_DIR to the directory containing nccl_device.h "
      "(for a source tree: <nccl>/src/include).")
  endif()
  if(NOT NCCL_PUBLIC_INCLUDE_DIR)
    get_filename_component(_NCCL_SRC_ROOT "${NCCL_DEVICE_INCLUDE_DIR}/../.." ABSOLUTE)
    if(EXISTS "${_NCCL_SRC_ROOT}/build/local/include/nccl.h")
      set(NCCL_PUBLIC_INCLUDE_DIR "${_NCCL_SRC_ROOT}/build/local/include")
    endif()
  endif()
  if(NOT EXISTS "${NCCL_PUBLIC_INCLUDE_DIR}/nccl.h")
    message(FATAL_ERROR
      "ENABLE_GIN_AG requires the matching NCCL public headers. Set "
      "NCCL_PUBLIC_INCLUDE_DIR to the directory containing nccl.h.")
  endif()
  # A source-tree nccl_device.h may enable the GDAKI backend on Hopper and
  # include DOCA GPUNetIO headers that live outside src/include. Installed NCCL
  # headers are self-contained; add the source auxiliary path only when present.
  get_filename_component(_NCCL_SRC_ROOT "${NCCL_DEVICE_INCLUDE_DIR}/../.." ABSOLUTE)
  set(NCCL_DEVICE_AUX_INCLUDE_DIR "")
  if(EXISTS "${_NCCL_SRC_ROOT}/src/transport/net_ib/gdaki/doca-gpunetio/include")
    set(NCCL_DEVICE_AUX_INCLUDE_DIR
        "${_NCCL_SRC_ROOT}/src/transport/net_ib/gdaki/doca-gpunetio/include")
  endif()
  get_filename_component(_NCCL_INSTALL_PREFIX "${NCCL_PUBLIC_INCLUDE_DIR}" DIRECTORY)
  find_library(NCCL_STATIC_LIBRARY
    NAMES nccl_static
    PATHS "${_NCCL_INSTALL_PREFIX}/lib"
    NO_DEFAULT_PATH)
  if(NOT NCCL_STATIC_LIBRARY)
    message(FATAL_ERROR
      "ENABLE_GIN_AG requires the matching libnccl_static.a under "
      "${_NCCL_INSTALL_PREFIX}/lib. Build the selected NCCL tree first.")
  endif()
  add_definitions(-DFLUX_ENABLE_GIN_AG)
  message(STATUS "GIN AG enabled; NCCL Device API headers: ${NCCL_DEVICE_INCLUDE_DIR}")
  message(STATUS "GIN AG NCCL static library: ${NCCL_STATIC_LIBRARY}")
endif()
message("PYTHONPATH: ${PYTHONPATH}")
message("NVShmem Support: ${ENABLE_NVSHMEM}")

# find cuda specify cuda path if other than default set(CUDA_TOOLKIT_ROOT_DIR
# /path/to/installed/cuda)
find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
if(CUDAToolkit_VERSION VERSION_LESS "11.0")
  message(FATAL_ERROR "requires cuda to be >= 11.0")
elseif(CUDAToolkit_VERSION VERSION_LESS "12.0")
  set(CUDAARCHS
      "80"
      CACHE STRING "CUDA Architectures")
elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.4")
  set(CUDAARCHS
      "80;89;90"
      CACHE STRING "CUDA Architectures")
else()
  set(CUDAARCHS
      "80;90"
      CACHE STRING "CUDA Architectures")
endif()

set(CMAKE_CUDA_ARCHITECTURES ${CUDAARCHS})
message(STATUS "CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

set(SM_CORES "")
option(GPU_SM_CORES "Manually specify GPU SM core number (e.g., '108;78'). Overrides auto-detection." "")

if(GPU_SM_CORES)
    set(SM_CORES ${GPU_SM_CORES})
    message(STATUS "Generator: Using manually specified GPU SM core number: ${SM_CORES}")
else()
    message(STATUS "Generator: GPU_SM_CORES not specified, attempting auto-detection by GPU name.")

    find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)

    if(NVIDIA_SMI_EXECUTABLE)
        execute_process(
            COMMAND ${NVIDIA_SMI_EXECUTABLE} --query-gpu=name --format=csv,noheader
            OUTPUT_VARIABLE DETECTED_GPU_NAMES
            RESULT_VARIABLE NVIDIA_SMI_RESULT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        if(NVIDIA_SMI_RESULT EQUAL 0 AND DETECTED_GPU_NAMES)
            string(REPLACE "\n" ";" GPU_NAME_LIST ${DETECTED_GPU_NAMES})
            set(SM_CORE_LIST "")

            foreach(GPU_NAME ${GPU_NAME_LIST})
                set(CURRENT_SM_COUNT 0)
                if(GPU_NAME MATCHES "A100")
                    set(CURRENT_SM_COUNT 108)
                elseif(GPU_NAME MATCHES "H800")
                    set(CURRENT_SM_COUNT 132)
                elseif(GPU_NAME MATCHES "L20")
                    set(CURRENT_SM_COUNT 92)
                elseif(GPU_NAME MATCHES "H20")
                    set(CURRENT_SM_COUNT 78)
                endif()

                if(NOT CURRENT_SM_COUNT EQUAL 0)
                    list(APPEND SM_CORE_LIST ${CURRENT_SM_COUNT})
                else()
                    message(WARNING "Generator: Could not determine SM count for GPU '${GPU_NAME}'. Please add it to the lookup list in CMakeLists.txt.")
                endif()
            endforeach()

            if(SM_CORE_LIST)
                list(REMOVE_DUPLICATES SM_CORE_LIST)
                list(JOIN SM_CORE_LIST ";" SM_CORES)
                message(STATUS "Generator: Auto-detected SM cores from GPU names: ${SM_CORES}")
            endif()
        else()
            message(WARNING "Generator: nvidia-smi command failed or no GPUs detected. SM core filtering will be disabled.")
        endif()
    else()
        message(WARNING "Generator: nvidia-smi not found. SM core filtering will be disabled.")
    endif()
endif()


set(CUDA_ARCH_FLAGS)
foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
  list(APPEND CUDA_ARCH_FLAGS
       "-gencode=arch=compute_${ARCH},code=\\\"sm_${ARCH},compute_${ARCH}\\\"")
endforeach()

string(JOIN " " JOINED_CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${JOINED_CUDA_ARCH_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wno-psabi")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler=-fno-strict-aliasing")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -DCUTLASS_DEBUG_TRACE_LEVEL=0")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

set(CMAKE_CXX_STANDARD "17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")

if(CUTLASS_TRACE)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUTLASS_DEBUG_TRACE_LEVEL=1")
endif()

if(FLUX_DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFLUX_DEBUG")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DFLUX_DEBUG")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(BUILD_THS)
  message("Build THS on")
  find_package(Python 3 REQUIRED)
  find_program(PYTHON_EXECUTABLE NAMES python3 python)

  set(TORCH_CUDA_ARCH_LIST)
  foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
    if(ARCH STREQUAL "80")
      list(APPEND TORCH_CUDA_ARCH_LIST "8.0")
    elseif(ARCH STREQUAL "89")
      list(APPEND TORCH_CUDA_ARCH_LIST "8.9")
    elseif(ARCH STREQUAL "90")
      list(APPEND TORCH_CUDA_ARCH_LIST "9.0")
    else()
      message(WARNING "Unsupported CUDA arch [${ARCH}] for TORCH_CUDA_ARCH")
    endif()
  endforeach()

  # to make old version pytorch happy
  string(JOIN ";" TORCH_CUDA_ARCH_LIST_STR ${TORCH_CUDA_ARCH_LIST})
  set(ENV{TORCH_CUDA_ARCH_LIST} "${TORCH_CUDA_ARCH_LIST_STR}")

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} "-c"
            "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE TORCH_DIR)
  if(NOT _PYTHON_SUCCESS MATCHES 0)
    message("PY:${PYTHONPATH}")
    message(FATAL_ERROR "Torch config Error.")
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})

  # CUDA 13 no longer provides the legacy nvToolsExt library target, while
  # PyTorch 2.6 still references CUDA::nvToolsExt from its CMake package.
  # CMake's FindCUDAToolkit exposes header-only CUDA::nvtx3, so provide a
  # compatibility alias before importing Torch.
  if(TARGET CUDA::nvtx3 AND NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
    set_property(TARGET CUDA::nvToolsExt PROPERTY
                 INTERFACE_LINK_LIBRARIES CUDA::nvtx3)
    message(STATUS "CUDA compatibility: CUDA::nvToolsExt -> CUDA::nvtx3")
  endif()

  find_package(Torch REQUIRED)
  find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_DIR}/lib")

  if(TORCH_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  endif()

  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} "-c"
      "from __future__ import print_function; from distutils import sysconfig;
print(sysconfig.get_python_inc());
print(sysconfig.get_config_var('EXT_SUFFIX'));"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE _PYTHON_VALUES)
  if(NOT _PYTHON_SUCCESS MATCHES 0)
    message("PY:${PYTHON_EXECUTABLE}")
    message(FATAL_ERROR "Python config Error.")
  endif()
  string(REGEX REPLACE ";" "\\\\;" _PYTHON_VALUES ${_PYTHON_VALUES})
  string(REGEX REPLACE "\n" ";" _PYTHON_VALUES ${_PYTHON_VALUES})
  list(GET _PYTHON_VALUES 0 PY_INCLUDE_DIR)
  list(GET _PYTHON_VALUES 1 PY_SUFFIX)
  list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})

  execute_process(
    COMMAND
      ${PYTHON_EXECUTABLE} "-c"
      "from torch.utils import cpp_extension; import re; import torch;                                    \
                   version = tuple(int(i) for i in re.match('(\\d+)\\.(\\d+)\\.(\\d+)', torch.__version__).groups()); \
                   args = ([],True,False,False) if version >= (1, 8, 0) else ([],True,False);                         \
                   print(' '.join(cpp_extension._prepare_ldflags(*args)),end='');"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE TORCH_LINK)
  message("-- TORCH_LINK ${TORCH_LINK}")
  if(NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "PyTorch link config Error.")
  endif()

  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import torch; _ = torch.float8_e4m3fn"
    RESULT_VARIABLE TORCH_FP8_STATUS
    OUTPUT_QUIET ERROR_QUIET)

  if(TORCH_FP8_STATUS EQUAL 0)
    message(STATUS "PyTorch supports FP8")
    add_compile_options("-DTORCH_SUPPORT_FP8")
  else()
    message(WARNING "PyTorch does not support FP8")
  endif()

endif()

# force use sm90a for cutlass
string(REGEX REPLACE "sm_90" "sm_90a" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
string(REGEX REPLACE "compute_90" "compute_90a" CMAKE_CUDA_FLAGS
                     ${CMAKE_CUDA_FLAGS})

set(COMMON_HEADER_DIRS
    ${NCCL_ROOT}/include
    ${NCCL_ROOT}/include/nccl/detail/include
    ${PROJECT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include
    ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
    ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/library/include
    ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/profiler/include)
if(ENABLE_GIN_AG)
  list(APPEND COMMON_HEADER_DIRS
       ${NCCL_DEVICE_INCLUDE_DIR} ${NCCL_PUBLIC_INCLUDE_DIR} ${NCCL_DEVICE_AUX_INCLUDE_DIR})
endif()

set(COMMON_LIB_DIRS "")
list(APPEND COMMON_LIB_DIRS "${NCCL_ROOT}/lib")
list(APPEND COMMON_LIB_DIRS "${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "NCCL root dir: ${NCCL_ROOT}")
message(ENABLE_NVSHMEM "ENABLE_NVSHMEM is set to: ${ENABLE_NVSHMEM}")
if(ENABLE_NVSHMEM)
  add_definitions(-DFLUX_SHM_USE_NVSHMEM)
  message(STATUS "NVSHMEM home dir: ${NVSHMEM_HOME}")
  if(NOT EXISTS ${NVSHMEM_HOME})
    message(
      FATAL_ERROR "NVSHMEM not found. Please run `pip3 install nvidia-nvshmem-cu12` first.")
  endif()
  list(APPEND COMMON_HEADER_DIRS "${NVSHMEM_HOME}/include")
  list(APPEND COMMON_LIB_DIRS "${NVSHMEM_HOME}/lib")
endif()

# append headers explicitly for .cu files, in order to enable vscode clangd
# intellisense
foreach(inc_dir ${COMMON_HEADER_DIRS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -I${inc_dir}")
endforeach()
message("final CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

include_directories(${COMMON_HEADER_DIRS})

link_directories(${COMMON_LIB_DIRS})

if(WITH_PROTOBUF)
  find_package(Protobuf REQUIRED)
  add_subdirectory(proto)
endif()

add_subdirectory(src)
if(BUILD_TEST)
  add_subdirectory(test)
endif()
