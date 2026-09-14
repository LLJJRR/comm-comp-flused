#!/bin/bash
set -e
set -x

########################################
# CONFIG
########################################
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FLUX_DIR=${FLUX_DIR:-${SCRIPT_DIR}}
VENV_DIR=${VENV_DIR:-/root/venv-flux}
NCCL_SOURCE_ROOT=${NCCL_SOURCE_ROOT:-${FLUX_DIR}/3rdparty/nccl}
NCCL_INSTALL_ROOT=${NCCL_INSTALL_ROOT:-${NCCL_SOURCE_ROOT}/build/local}
JOBS=${JOBS:-16}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
CLEAN_BUILD=${CLEAN_BUILD:-0}
CLEAN_NCCL=${CLEAN_NCCL:-0}
RUN_VERIFY=${RUN_VERIFY:-0}
RUN_GEMM_TEST=${RUN_GEMM_TEST:-0}
GIN_MODE=${GIN_MODE:-both}   # ag | rs | both
PYTHON_BIN=${PYTHON_BIN:-python3}

########################################
# CHECK / PYTHON ENV
########################################
if [ ! -d "${FLUX_DIR}" ]; then
    echo "ERROR: FLUX_DIR=${FLUX_DIR} does not exist"
    exit 1
fi
if [ ! -f "${FLUX_DIR}/3rdparty/cutlass/include/cutlass/cutlass.h" ]; then
    echo "ERROR: CUTLASS is missing: ${FLUX_DIR}/3rdparty/cutlass/include/cutlass/cutlass.h"
    exit 1
fi
if [ ! -f "${NCCL_SOURCE_ROOT}/Makefile" ] || [ ! -f "${NCCL_SOURCE_ROOT}/src/include/nccl_device.h" ]; then
    echo "ERROR: vendored NCCL source is incomplete: ${NCCL_SOURCE_ROOT}"
    exit 1
fi

if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Using active virtualenv: ${VIRTUAL_ENV}"
elif [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "No virtualenv found; using current Python: $(command -v ${PYTHON_BIN})"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: PYTHON_BIN=${PYTHON_BIN} not found"
    exit 1
fi

case "${GIN_MODE}" in
    ag)   GIN_BUILD_ARGS=(--gin-ag) ;;
    rs)   GIN_BUILD_ARGS=(--gin-rs) ;;
    both) GIN_BUILD_ARGS=(--gin-ag --gin-rs) ;;
    *)
        echo "ERROR: GIN_MODE must be ag, rs, or both; got '${GIN_MODE}'"
        exit 2
        ;;
esac

export CUDA_HOME
export PATH=${CUDA_HOME}/bin:$PATH

export NVSHMEM_HOME=$(${PYTHON_BIN} - <<'PY'
import pathlib
import nvidia.nvshmem
print(pathlib.Path(nvidia.nvshmem.__path__[0]))
PY
)

echo "NVSHMEM_HOME=${NVSHMEM_HOME}"

TORCH_CUDA_VERSION=$(${PYTHON_BIN} - <<'PY'
import torch
print(torch.version.cuda)
PY
)
NVCC_CUDA_VERSION=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)

echo "CUDA_HOME=${CUDA_HOME}"
echo "torch CUDA=${TORCH_CUDA_VERSION}"
echo "nvcc CUDA=${NVCC_CUDA_VERSION}"

if [ -z "${NVCC_CUDA_VERSION}" ]; then
    echo "ERROR: failed to parse nvcc CUDA version from ${CUDA_HOME}/bin/nvcc"
    exit 1
fi
if [ "${TORCH_CUDA_VERSION}" != "${NVCC_CUDA_VERSION}" ]; then
    echo "ERROR: CUDA toolkit version must match PyTorch CUDA version"
    echo "torch=${TORCH_CUDA_VERSION}, nvcc=${NVCC_CUDA_VERSION}"
    exit 1
fi

cd "${NVSHMEM_HOME}/lib"
ln -sf libnvshmem_host.so.3 libnvshmem_host.so

########################################
# BUILD ENV
########################################
cd "${FLUX_DIR}"
export NVSHMEM_HOME
export NCCL_SOURCE_ROOT
export NCCL_INSTALL_ROOT
export NCCL_ROOT=${NCCL_INSTALL_ROOT}
export FLUX_SHM_USE_NVSHMEM=1
export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

function write_flux_env_script() {
    local activate_line=""
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -f "${VIRTUAL_ENV}/bin/activate" ]; then
        activate_line="source ${VIRTUAL_ENV}/bin/activate"
    elif [ -f "${VENV_DIR}/bin/activate" ]; then
        activate_line="source ${VENV_DIR}/bin/activate"
    fi

    cat >"${FLUX_DIR}/flux_env.sh" <<ENVEOF
${activate_line}
export CUDA_HOME=${CUDA_HOME}
export PATH=${CUDA_HOME}/bin:\$PATH
export NVSHMEM_HOME=${NVSHMEM_HOME}
export NCCL_SOURCE_ROOT=${NCCL_SOURCE_ROOT}
export NCCL_INSTALL_ROOT=${NCCL_INSTALL_ROOT}
export NCCL_ROOT=${NCCL_INSTALL_ROOT}
export NCCL_DEVICE_INCLUDE_DIR=${NCCL_SOURCE_ROOT}/src/include
export NCCL_PUBLIC_INCLUDE_DIR=${NCCL_INSTALL_ROOT}/include
export PYTHONPATH=${FLUX_DIR}/python:\${PYTHONPATH:-}
export LD_LIBRARY_PATH=${FLUX_DIR}/python/flux/lib:${NVSHMEM_HOME}/lib:${NCCL_INSTALL_ROOT}/lib:${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}
export FLUX_SHM_USE_NVSHMEM=1
ENVEOF

    if [ "${GIN_MODE}" = "ag" ] || [ "${GIN_MODE}" = "both" ]; then
        echo 'export FLUX_ENABLE_GIN_AG=1' >>"${FLUX_DIR}/flux_env.sh"
    fi
    if [ "${GIN_MODE}" = "rs" ] || [ "${GIN_MODE}" = "both" ]; then
        echo 'export FLUX_ENABLE_GIN_RS=1' >>"${FLUX_DIR}/flux_env.sh"
    fi
}


########################################
# OPTIONAL CLEAN
########################################
# Normal git-clone development uses incremental builds.  Clean only when
# explicitly requested, e.g. CLEAN_BUILD=1 CLEAN_NCCL=1 ./build_h100_gin.sh.
if [ "${CLEAN_BUILD}" = "1" ]; then
    rm -rf build
    rm -rf build/lib.*
    rm -f python/flux_ths_pybind*.so
fi
if [ "${CLEAN_NCCL}" = "1" ]; then
    rm -rf "${NCCL_SOURCE_ROOT}/build"
fi

########################################
# BUILD
########################################
./build.sh \
    "${GIN_BUILD_ARGS[@]}" \
    --runnable \
    --arch 90 \
    --sm-cores 132 \
    --nvshmem \
    --jobs "${JOBS}"

########################################
# RUNTIME ENV + OPTIONAL VERIFY
########################################
# Keep the known-good runtime exports, but do not make build success depend on
# launching a GPU test.  flux_env.sh is machine-local and ignored by git.
write_flux_env_script

if [ "${RUN_VERIFY}" = "1" ] || [ "${RUN_GEMM_TEST}" = "1" ]; then
    # shellcheck disable=SC1091
    source "${FLUX_DIR}/flux_env.sh"
fi

if [ "${RUN_VERIFY}" = "1" ]; then
GIN_MODE="${GIN_MODE}" ${PYTHON_BIN} - <<'PY'
import os
import torch
import flux
from flux.cpp_mod import NotCompiled

mode = os.environ["GIN_MODE"]
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("SM count:", torch.cuda.get_device_properties(0).multi_processor_count)
if mode in ("ag", "both"):
    assert not isinstance(flux.GinAGKernel, NotCompiled), "GinAGKernel was not compiled"
    print("GinAGKernel: compiled")
if mode in ("rs", "both"):
    assert not isinstance(flux.GemmRS, NotCompiled), "GemmRS was not compiled"
    print("GemmRS (GIN RS enabled build): compiled")
print("flux import ok")
PY
fi

if [ "${RUN_GEMM_TEST}" = "1" ]; then
    echo "ERROR: RUN_GEMM_TEST belongs to the full/comm_none build and is intentionally excluded from the narrow runnable GIN build."
    echo "Use RUN_VERIFY=1 here; run test_gin_* on a supported multi-node setup."
    exit 2
fi

echo ""
echo "===================================="
echo "FLUX GIN BUILD SUCCESS (mode=${GIN_MODE}, narrow runnable profile)"
echo "runtime env: source ${FLUX_DIR}/flux_env.sh"
echo "NCCL_SOURCE_ROOT=${NCCL_SOURCE_ROOT}"
echo "incremental build: CLEAN_BUILD=0 CLEAN_NCCL=0"
echo "===================================="
