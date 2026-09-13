cmake_minimum_required(VERSION 3.17)

set(OP_REGS "")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/registers")
execute_process(
  COMMAND ${FLUX_GENERATOR_BINARY_DIR}/gen_ag_gemm "--dir=./registers" "--archs=${CUDAARCHS}" "--sm_cores=${SM_CORES}"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND_ERROR_IS_FATAL ANY
  COMMAND_ECHO STDOUT
)

file(GLOB OP_REGS ${CMAKE_CURRENT_BINARY_DIR}/registers/*.cu)
file(GLOB TUNING_CONFIGS tuning_config/*.cu)
set(CU_FILES
  ${OP_REGS}
  ${TUNING_CONFIGS}
)
if(ENABLE_GIN_AG)
  list(APPEND CU_FILES gin/gin_ag_gemm_comm.cu)
endif()

set(LIB_NAME "flux_cuda_all_gather")
flux_add_op_cu_obj_lib(${LIB_NAME} "${CU_FILES}")
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
if(ENABLE_GIN_AG)
  # NCCL's Device API GIN/barrier implementation is gated by this Linux build
  # define in nccl_device.h/impl headers. NCCL's own device target and examples
  # set the same definition.
  target_compile_definitions(${LIB_NAME} PRIVATE NCCL_OS_LINUX)
endif()

set(FLUX_CUDA_OP_TARGETS ${FLUX_CUDA_OP_TARGETS})
list(APPEND FLUX_CUDA_OP_TARGETS ${LIB_NAME})
set(FLUX_CUDA_OP_TARGETS ${FLUX_CUDA_OP_TARGETS} PARENT_SCOPE)

if (BUILD_THS)
  file(GLOB THS_FILES ths_op/*.cc)
  set(FLUX_THS_OP_FILES ${FLUX_THS_OP_FILES})
  list(APPEND FLUX_THS_OP_FILES ${THS_FILES})
  set(FLUX_THS_OP_FILES ${FLUX_THS_OP_FILES} PARENT_SCOPE)
  flux_add_ths_op_target("ag_gemm")
endif()
