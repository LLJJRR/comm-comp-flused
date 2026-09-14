cmake_minimum_required(VERSION 3.17)

set(GEMM_OP_REG_INP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cuda/gemm_op_registry.cu.in)

# append 'flux/src/' to include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# create an object library for each op subfolder
function(flux_add_op_cu_obj_lib TARGET FILES)
  add_library(${TARGET} OBJECT ${FILES})
  set_target_properties(${TARGET} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(${TARGET} PUBLIC CUDA::cudart ${CMAKE_THREAD_LIBS_INIT} )
  set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
endfunction()

set_property(GLOBAL PROPERTY FLUX_THS_TARGETS)
function (flux_add_ths_op_target TARGET)
  get_property(FLUX_THS_TARGETS_LOCAL GLOBAL PROPERTY FLUX_THS_TARGETS)
  set_property(GLOBAL PROPERTY FLUX_THS_TARGETS "${TARGET};${FLUX_THS_TARGETS_LOCAL}")
endfunction()

set(FLUX_CUDA_OP_TARGETS "") # cu op targets of subdirectory
if (BUILD_THS)
  set(FLUX_THS_OP_FILES "") # ths op source files of subdirectory
endif()

include(ExternalProject)
ExternalProject_Add(
  FLUX_GENERATOR
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generator
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""  # Prevents automatic build during build phase
  INSTALL_COMMAND ""  # Prevents automatic install
)
ExternalProject_Get_property(FLUX_GENERATOR SOURCE_DIR)
ExternalProject_Get_property(FLUX_GENERATOR BINARY_DIR)
set(FLUX_GENERATOR_BINARY_DIR ${BINARY_DIR})
set(FLUX_GENERATOR_SOURCE_DIR ${SOURCE_DIR})

execute_process(
  COMMAND ${CMAKE_COMMAND} ${FLUX_GENERATOR_SOURCE_DIR}
  WORKING_DIRECTORY ${FLUX_GENERATOR_BINARY_DIR}
  COMMAND_ERROR_IS_FATAL ANY
)

set(_FLUX_GENERATOR_TARGETS "")
if(FLUX_BUILD_SCOPE STREQUAL "comm_none")
  list(APPEND _FLUX_GENERATOR_TARGETS gen_comm_none)
elseif(FLUX_BUILD_SCOPE STREQUAL "ag_gemm")
  list(APPEND _FLUX_GENERATOR_TARGETS gen_ag_gemm)
elseif(FLUX_BUILD_SCOPE STREQUAL "gemm_rs")
  list(APPEND _FLUX_GENERATOR_TARGETS gen_gemm_rs)
elseif(FLUX_BUILD_SCOPE STREQUAL "gin_fusions")
  list(APPEND _FLUX_GENERATOR_TARGETS gen_ag_gemm gen_gemm_rs)
elseif(FLUX_BUILD_SCOPE STREQUAL "gemm_a2a_transpose")
  list(APPEND _FLUX_GENERATOR_TARGETS gen_gemm_a2a_transpose)
elseif(FLUX_BUILD_SCOPE STREQUAL "a2a_transpose_gemm")
  list(APPEND _FLUX_GENERATOR_TARGETS gen_a2a_transpose_gemm)
elseif(FLUX_BUILD_SCOPE STREQUAL "moe_ag_scatter")
  list(APPEND _FLUX_GENERATOR_TARGETS gen_moe_ag_scatter)
elseif(FLUX_BUILD_SCOPE STREQUAL "moe_gather_rs")
  list(APPEND _FLUX_GENERATOR_TARGETS gen_moe_gather_rs)
elseif(NOT FLUX_BUILD_SCOPE STREQUAL "all")
  message(FATAL_ERROR "Unknown FLUX_BUILD_SCOPE='${FLUX_BUILD_SCOPE}'")
endif()

if(_FLUX_GENERATOR_TARGETS)
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build . --parallel 8 --target ${_FLUX_GENERATOR_TARGETS}
    WORKING_DIRECTORY ${FLUX_GENERATOR_BINARY_DIR}
    COMMAND_ERROR_IS_FATAL ANY
  )
else()
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build . --parallel 8
    WORKING_DIRECTORY ${FLUX_GENERATOR_BINARY_DIR}
    COMMAND_ERROR_IS_FATAL ANY
  )
endif()

if (WITH_TRITON_AOT)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import triton; from packaging.version import Version; assert Version(triton.__version__) >= Version('3.0.0')"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND_ERROR_IS_FATAL ANY
  )
  set(MOE_AG_SCATTER_KERNEL ${PROJECT_SOURCE_DIR}/python/flux_triton/kernels/moe_ag_scatter.py:moe_ag_scatter_grouped_gemm_kernel ${PROJECT_SOURCE_DIR}/python/flux_triton/kernels/moe_gather_rs.py:moe_gather_rs_grouped_gemm_kernel)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/python/flux_triton/tools/compile_aot.py --workspace ${CMAKE_CURRENT_SOURCE_DIR}/triton_aot_generated --kernels ${MOE_AG_SCATTER_KERNEL}  --library flux_triton_aot
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND_ERROR_IS_FATAL ANY
  )
  add_subdirectory(triton_aot_generated)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/triton_aot_generated)
  add_definitions(-DFLUX_WITH_TRITON_AOT)
endif()

# A minimal scope intentionally configures/builds only one fusion family.  It is
# for fast kernel iteration and therefore does not build the aggregate CUDA/THS
# libraries.  The default "all" scope retains the original full build behavior.
if(FLUX_BUILD_SCOPE STREQUAL "all")
  add_subdirectory(comm_none)
  add_subdirectory(coll)
  add_subdirectory(ag_gemm)
  add_subdirectory(gemm_rs)
  add_subdirectory(gemm_a2a_transpose)
  add_subdirectory(a2a_transpose_gemm)
  add_subdirectory(inplace_cast)

  find_package(CUDAToolkit REQUIRED)
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.3")
    add_subdirectory(quantization)
  endif()

  if(ENABLE_NVSHMEM)
    add_subdirectory(moe_ag_scatter)
    add_subdirectory(moe_gather_rs)
  endif()
elseif(FLUX_BUILD_SCOPE STREQUAL "comm_none")
  add_subdirectory(comm_none)
elseif(FLUX_BUILD_SCOPE STREQUAL "ag_gemm")
  # Runnable AG needs the small coll support target because the regular
  # AGKernel THS implementation owns an AllGatherOp. Compile-check mode does
  # not link THS and can stay at the single AG object library.
  if(FLUX_BUILD_RUNNABLE)
    add_subdirectory(coll)
  endif()
  add_subdirectory(ag_gemm)
elseif(FLUX_BUILD_SCOPE STREQUAL "gemm_rs")
  if(FLUX_BUILD_RUNNABLE)
    add_subdirectory(coll)
  endif()
  add_subdirectory(gemm_rs)
elseif(FLUX_BUILD_SCOPE STREQUAL "gin_fusions")
  if(NOT ENABLE_GIN_AG OR NOT ENABLE_GIN_RS)
    message(FATAL_ERROR "gin_fusions scope requires ENABLE_GIN_AG=ON and ENABLE_GIN_RS=ON")
  endif()
  add_subdirectory(coll)
  add_subdirectory(ag_gemm)
  add_subdirectory(gemm_rs)
elseif(FLUX_BUILD_SCOPE STREQUAL "gemm_a2a_transpose")
  add_subdirectory(gemm_a2a_transpose)
elseif(FLUX_BUILD_SCOPE STREQUAL "a2a_transpose_gemm")
  add_subdirectory(a2a_transpose_gemm)
elseif(FLUX_BUILD_SCOPE STREQUAL "moe_ag_scatter")
  if(NOT ENABLE_NVSHMEM)
    message(FATAL_ERROR "moe_ag_scatter minimal build requires ENABLE_NVSHMEM=ON")
  endif()
  add_subdirectory(moe_ag_scatter)
elseif(FLUX_BUILD_SCOPE STREQUAL "moe_gather_rs")
  if(NOT ENABLE_NVSHMEM)
    message(FATAL_ERROR "moe_gather_rs minimal build requires ENABLE_NVSHMEM=ON")
  endif()
  add_subdirectory(moe_gather_rs)
endif()
message(STATUS "FLUX_BUILD_SCOPE=${FLUX_BUILD_SCOPE}")
message(STATUS "FLUX_BUILD_RUNNABLE=${FLUX_BUILD_RUNNABLE}")
message(STATUS "op_targets: ${FLUX_CUDA_OP_TARGETS}")

if(FLUX_BUILD_SCOPE STREQUAL "all" OR FLUX_BUILD_RUNNABLE)
  # This must be after the selected op subdirectories so the aggregate core
  # and THS libraries link exactly the requested operator family.
  add_subdirectory(cuda)

  if (BUILD_THS)
    add_subdirectory(ths_op)
  endif()
endif()

include(GNUInstallDirs)
install(
  DIRECTORY ./ "${PROJECT_SOURCE_DIR}/include/flux/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flux/
  FILES_MATCHING PATTERN "*/ths_op/*.h")

file(GLOB FLUX_HEADERS ${PROJECT_SOURCE_DIR}/include/flux/*.h)
install(
  FILES ${FLUX_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/flux/
)
install(
  FILES ${PROJECT_SOURCE_DIR}/cmake/FluxConfig.cmake
  DESTINATION share/cmake/Flux/
)
