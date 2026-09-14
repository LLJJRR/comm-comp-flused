#!/bin/bash
set -e
set -x

VENV_DIR=${VENV_DIR:-/root/venv-flux}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
PYTHON_BIN=${PYTHON_BIN:-python3}
TORCH_VERSION=${TORCH_VERSION:-}

if [ "${EUID}" -eq 0 ]; then
    APT=(apt)
elif command -v sudo >/dev/null 2>&1; then
    APT=(sudo apt)
else
    echo "ERROR: apt package installation requires root or sudo"
    exit 1
fi

"${APT[@]}" update
"${APT[@]}" install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    python3 \
    python3-dev \
    python3-venv

# NCCL GIN / NET-IB requires verbs and mlx5dv development headers. Most H100
# cloud images already provide MLNX_OFED/DOCA. Do not replace that RDMA stack
# when the headers already exist; install distro development packages only when
# they are missing.
if [ ! -f /usr/include/infiniband/verbs.h ] || [ ! -f /usr/include/infiniband/mlx5dv.h ]; then
    "${APT[@]}" install -y \
        libibverbs-dev \
        librdmacm-dev \
        libnl-3-dev \
        libnl-route-3-dev
fi

if [ ! -f /usr/include/infiniband/verbs.h ] || [ ! -f /usr/include/infiniband/mlx5dv.h ]; then
    echo "ERROR: RDMA development headers are still missing"
    echo "required: /usr/include/infiniband/verbs.h"
    echo "required: /usr/include/infiniband/mlx5dv.h"
    echo "If this image uses MLNX_OFED/DOCA, install its matching development packages instead of replacing the driver stack."
    exit 1
fi

if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at ${CUDA_HOME}/bin/nvcc"
    exit 1
fi
NVCC_CUDA_VERSION=$(${CUDA_HOME}/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
if [ -z "${NVCC_CUDA_VERSION}" ]; then
    echo "ERROR: failed to parse CUDA version from nvcc"
    exit 1
fi
CUDA_WHEEL_TAG=cu${NVCC_CUDA_VERSION/./}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${CUDA_WHEEL_TAG}}

if [ ! -d "${VENV_DIR}" ]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install packaging ninja numpy

# Never silently replace a matching CUDA/PyTorch environment with a hard-coded
# older wheel. Keep an existing match; otherwise install from the wheel index
# corresponding to the selected nvcc (for example CUDA 13.0 -> cu130).
TORCH_CUDA_VERSION=$(python - <<'PY' 2>/dev/null || true
try:
    import torch
    print(torch.version.cuda or "")
except Exception:
    pass
PY
)
if [ "${TORCH_CUDA_VERSION}" != "${NVCC_CUDA_VERSION}" ]; then
    TORCH_SPEC=torch
    if [ -n "${TORCH_VERSION}" ]; then
        TORCH_SPEC="torch==${TORCH_VERSION}"
    fi
    python -m pip install \
        "${TORCH_SPEC}" \
        torchvision \
        torchaudio \
        --index-url "${TORCH_INDEX_URL}"
fi

TORCH_CUDA_VERSION=$(python - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)
if [ "${TORCH_CUDA_VERSION}" != "${NVCC_CUDA_VERSION}" ]; then
    echo "ERROR: PyTorch CUDA (${TORCH_CUDA_VERSION}) does not match nvcc (${NVCC_CUDA_VERSION})"
    echo "Set TORCH_VERSION/TORCH_INDEX_URL explicitly if this CUDA wheel is not available from the default index."
    exit 1
fi

# Preserve an already working NVSHMEM Python package. The project currently
# supports the validated nvidia-nvshmem-cu12 package as a fallback even when the
# host toolkit is CUDA 13, so do not uninstall/reinstall it unnecessarily.
if ! python - <<'PY'
import nvidia.nvshmem
PY
then
    python -m pip install nvidia-nvshmem-cu12==3.3.9
fi

export NVSHMEM_HOME=$(python - <<'PY'
import pathlib
import nvidia.nvshmem
print(pathlib.Path(nvidia.nvshmem.__path__[0]))
PY
)
cd "${NVSHMEM_HOME}/lib"
ln -sf libnvshmem_host.so.3 libnvshmem_host.so

echo ""
echo "===================================="
echo "H100 ENV PREPARE SUCCESS"
echo "VENV_DIR=${VENV_DIR}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "NVSHMEM_HOME=${NVSHMEM_HOME}"
echo "PyTorch=$(python -c 'import torch; print(torch.__version__)')"
echo "Torch CUDA=$(python -c 'import torch; print(torch.version.cuda)')"
echo "===================================="
