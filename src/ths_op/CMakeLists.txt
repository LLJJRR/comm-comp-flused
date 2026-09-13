cmake_minimum_required(VERSION 3.17)

file(GLOB OTHER_THS_FILES *.cc)
set(FLUX_THS_FILES
    ${FLUX_THS_OP_FILES}
    ${OTHER_THS_FILES}
)

message(STATUS "ths_files: ${FLUX_THS_FILES}")

add_library(flux_cuda_ths_op SHARED ${FLUX_THS_FILES})
if (WITH_TRITON_AOT)
    SET(TRITON_AOT_LIB "flux_triton_aot")
else()
    SET(TRITON_AOT_LIB )
endif()
target_link_libraries(flux_cuda_ths_op
    flux_cuda
    ${TORCH_LIBRARIES} ${TRITON_AOT_LIB})
if(ENABLE_GIN_AG)
  # cpp_mod.py preloads libflux_cuda_ths_op.so before importing the pybind
  # extension. Resolve the NCCL 2.30.x Device API symbols here against the
  # exact patched static library instead of relying on whichever NCCL PyTorch
  # may already have loaded in the process.
  target_link_libraries(flux_cuda_ths_op
    ${NCCL_STATIC_LIBRARY} pthread rt dl)
  target_link_options(flux_cuda_ths_op
    PRIVATE $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs=libnccl_static.a>)
endif()

# Write the unchached variable to a file
get_property(FLUX_THS_TARGETS GLOBAL PROPERTY FLUX_THS_TARGETS)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/flux_ths_targets.txt" "FLUX_THS_TARGETS=ths_op;${FLUX_THS_TARGETS}\n")

install(TARGETS flux_cuda_ths_op
    PUBLIC_HEADER DESTINATION include
)
