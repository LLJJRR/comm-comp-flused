#!/bin/bash
set -e
set -x

VENV_DIR=${VENV_DIR:-/root/venv-flux}

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
    python3.11 \
    python3.11-dev \
    python3.11-venv

# NCCL GIN / NET-IB requires verbs and mlx5dv development headers. Most H100
# cloud images already provide MLNX_OFED/DOCA. Do not replace that RDMA stack
# when the headers already exist; install distro development packages only when
# they are missing.
if [ ! -f /usr/include/infiniband/verbs.h ] || [ ! -f /usr/include/infiniband/mlx5dv.h ]; then
    apt install -y \
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

if [ ! -d "${VENV_DIR}" ]; then
    python3.11 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

python -m pip install -U pip setuptools wheel
python -m pip install packaging ninja

# Keep the same H100 environment that was already validated in the previous
# Flux fusion work. Change these only when the target image requires it.
python -m pip install \
    torch==2.6.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

python -m pip uninstall -y nvidia-nvshmem-cu12 || true
python -m pip install nvidia-nvshmem-cu12==3.3.9

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
echo "NVSHMEM_HOME=${NVSHMEM_HOME}"
echo "PyTorch=$(python -c 'import torch; print(torch.__version__)')"
echo "Torch CUDA=$(python -c 'import torch; print(torch.version.cuda)')"
echo "===================================="
