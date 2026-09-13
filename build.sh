#!/bin/bash
set -x
set -e

## Keep the CUDA toolkit selected by the caller. This is important on hosts
## with multiple CUDA installations because PyTorch and nvcc must match.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:$PATH
CMAKE=${CMAKE:-cmake}

ARCH=""
SM_CORES=""
BUILD_TEST="ON"
BDIST_WHEEL="OFF"
WITH_PROTOBUF="OFF"
FLUX_DEBUG="OFF"
ENABLE_NVSHMEM="OFF"
ENABLE_GIN_AG="OFF"
WITH_TRITON_AOT="OFF"

function clean_py() {
    rm -rf build/lib.*
    rm -rf python/lib
    rm -rf .egg/
    rm -rf python/flux.egg-info
    rm -rf python/flux_ths_pybind.*
}

function clean_all() {
    clean_py
    rm -rf build/
    rm -rf 3rdparty/nccl/build
    rm -rf 3rdparty/protobuf/build
}

# Iterate over the command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --arch)
        # Process the arch argument
        ARCH="$2"
        shift # Skip the argument value
        shift # Skip the argument key
        ;;
    --sm-cores)
        # Process the sm-cores argument
        SM_CORES="$2"
        shift # Skip the argument value
        shift # Skip the argument key
        ;;
    --no_test)
        BUILD_TEST="OFF"
        shift # Skip the argument value
        ;;
    --jobs)
        # Process the jobs argument
        JOBS="$2"
        shift # Skip the argument value
        shift # Skip the argument key
        ;;
    --clean-py)
        clean_py
        exit 0
        ;;
    --clean-all)
        clean_all
        exit 0
        ;;
    --debug)
        FLUX_DEBUG="ON"
        shift
        ;;
    --package)
        BDIST_WHEEL="ON"
        shift # Skip the argument key
        ;;
    --protobuf)
        WITH_PROTOBUF="ON"
        shift
        ;;
    --nvshmem)
        ENABLE_NVSHMEM="ON"
        shift
        ;;
    --gin-ag)
        ENABLE_GIN_AG="ON"
        shift
        ;;
    --triton-aot)
        WITH_TRITON_AOT="ON"
        shift
        ;;
    *)
        # Unknown argument
        echo "Unknown argument: $1"
        shift # Skip the argument
        ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=${SCRIPT_DIR}
PROTOBUF_ROOT=$PROJECT_ROOT/3rdparty/protobuf
# GIN AG must compile and link against the exact NCCL source tree that carries
# the Device API/GIN implementation being tested.  Keep source and install roots
# separate so an external patched NCCL tree can be selected without copying it
# into Flux's submodule directory.
NCCL_SOURCE_ROOT=${NCCL_SOURCE_ROOT:-$PROJECT_ROOT/3rdparty/nccl}
NCCL_INSTALL_ROOT=${NCCL_INSTALL_ROOT:-${NCCL_SOURCE_ROOT}/build/local}

cd ${PROJECT_ROOT}

if [[ -n $ARCH ]]; then
    build_args=" --arch ${ARCH}"
fi

if [[ -z $JOBS ]]; then
    JOBS=$(nproc --ignore 2)
fi

##### build protobuf #####
function build_protobuf() {
    if [ $WITH_PROTOBUF == "ON" ]; then
        pushd $PROTOBUF_ROOT
        mkdir -p $PWD/build/local
        pushd build
        CXXFLAGS_EXTRA=""
        use_cxx11_abi=$(python3 -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)")
        if [ $use_cxx11_abi == "False" ]; then
            CXXFLAGS_EXTRA="-D_GLIBCXX_USE_CXX11_ABI=0"
        fi
        CFLAGS="-fPIC" CXXFLAGS="-fPIC ${CXXFLAGS_EXTRA}" cmake ../cmake \
            -Dprotobuf_BUILD_TESTS=OFF \
            -Dprotobuf_BUILD_SHARED_LIBS=OFF \
            -DCMAKE_INSTALL_PREFIX=$(realpath local)
        make -j$(nproc)
        make install
        popd
        popd
    fi
}

function build_nccl() {
    pushd $NCCL_SOURCE_ROOT
    export BUILDDIR=${NCCL_SOURCE_ROOT}/build
    export PREFIX=${NCCL_INSTALL_ROOT}

    if [[ -n $ARCH ]]; then
        NCCL_COMPILE_OPTIONS_ARCH="" # default none
        arch_list=()
        IFS=";" read -ra arch_list <<<"$ARCH"
        for arch in "${arch_list[@]}"; do
            NCCL_COMPILE_OPTIONS_ARCH="-gencode=arch=compute_${arch},code=sm_${arch} ${NCCL_COMPILE_OPTIONS_ARCH}"
        done
        make -j${JOBS} src.staticlib CUDARTLIB=cudart NVCC_GENCODE="${NCCL_COMPILE_OPTIONS_ARCH}" VERBOSE=1
    else
        make -j${JOBS} src.staticlib CUDARTLIB=cudart VERBOSE=1
    fi
    # only install static lib
    mkdir -p ${PREFIX}/lib
    cp -P -v ${BUILDDIR}/lib/lib* ${PREFIX}/lib/
    cp -P -v -r ${BUILDDIR}/include ${PREFIX}/
    popd
}

##### build flux_cuda #####
function build_flux_cuda() {
    mkdir -p build
    pushd build
    export LIBFLUX_PREFIX=${PROJECT_ROOT}/python/flux
    if [ ! -f CMakeCache.txt ] || [ -z ${FLUX_BUILD_SKIP_CMAKE} ]; then
        CMAKE_ARGS=(
            -DENABLE_NVSHMEM=${ENABLE_NVSHMEM}
            -DENABLE_GIN_AG=${ENABLE_GIN_AG}
            -DNCCL_ROOT=${NCCL_INSTALL_ROOT}
            -DNCCL_DEVICE_INCLUDE_DIR=${NCCL_SOURCE_ROOT}/src/include
            -DNCCL_PUBLIC_INCLUDE_DIR=${NCCL_INSTALL_ROOT}/include
            -DNVSHMEM_HOME=${NVSHMEM_HOME}
            -DCUDAARCHS=${ARCH}
            -DGPU_SM_CORES=${SM_CORES}
            -DCMAKE_EXPORT_COMPILE_COMMANDS=1
            -DBUILD_TEST=${BUILD_TEST}
            -DCMAKE_INSTALL_PREFIX=${LIBFLUX_PREFIX}
        )
        if [ $WITH_PROTOBUF == "ON" ]; then
            CMAKE_ARGS+=(
                -DWITH_PROTOBUF=ON
                -DProtobuf_ROOT=${PROTOBUF_ROOT}/build/local
                -DProtobuf_PROTOC_EXECUTABLE=${PROTOBUF_ROOT}/build/local/bin/protoc
            )
        fi
        if [ $FLUX_DEBUG == "ON" ]; then
            CMAKE_ARGS+=(
                -DFLUX_DEBUG=ON
            )
        fi
        if [ $WITH_TRITON_AOT == "ON" ]; then
            CMAKE_ARGS+=(
                -DWITH_TRITON_AOT=ON
            )
            export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/python
        fi
        ${CMAKE} .. ${CMAKE_ARGS[@]}
    fi
    make -j${JOBS} VERBOSE=1
    make install
    popd
}

function merge_compile_commands() {
    cd $SCRIPT_DIR
    local ths_build_ninja
    ths_build_ninja=$(find ./build -path './build/temp.*/build.ninja' -print -quit 2>/dev/null || true)
    if [[ -n "${ths_build_ninja}" ]] && command -v ninja >/dev/null 2>&1; then
        ninja -f "${ths_build_ninja}" -t compdb >build/compile_commands_ths_op.json
        cat >build/merge_compile_commands.py <<EOF
import json
with open("build/compile_commands.json") as f:
    cmds = json.load(f)
with open("build/compile_commands_ths_op.json") as f:
    cmds_ths_op = json.load(f)
with open("build/compile_commands.json", "w") as f:
    json.dump(cmds+cmds_ths_op, f, indent=2)
EOF

        python3 build/merge_compile_commands.py
        echo "merge compile_commands.json done"
    elif [[ -z "${ths_build_ninja}" ]]; then
        echo "skip merge_compile_commands: torch extension build.ninja not found"
    else
        echo "Ninja is not installed. Ninja is required for flux_ths_pybind's compile_commands.json. run 'pip3 install ninja'"
    fi
}

function build_flux_py {
    LIBDIR=${PROJECT_ROOT}/python/flux/lib
    mkdir -p ${LIBDIR}

    pushd ${LIBDIR}
    if [ $ENABLE_NVSHMEM == "ON" ]; then
        export FLUX_SHM_USE_NVSHMEM=1
    fi
    export NCCL_ROOT=${NCCL_INSTALL_ROOT}
    if [ $ENABLE_GIN_AG == "ON" ]; then
        export FLUX_ENABLE_GIN_AG=1
        export NCCL_DEVICE_INCLUDE_DIR=${NCCL_SOURCE_ROOT}/src/include
        export NCCL_PUBLIC_INCLUDE_DIR=${NCCL_INSTALL_ROOT}/include
    fi
    popd
    ##### build flux torch bindings #####
    # The H100 environment uses a venv. Avoid setup.py develop --user there:
    # it can escape the venv and PEP517/build-isolation can pull mismatched deps.
    PIP_INSTALL_ARGS=(install -e . --no-build-isolation)
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        PIP_INSTALL_ARGS+=(--user)
    fi
    MAX_JOBS=${JOBS} python3 -m pip "${PIP_INSTALL_ARGS[@]}"
    if [ $BDIST_WHEEL == "ON" ]; then
        MAX_JOBS=${JOBS} python3 setup.py bdist_wheel
    fi
}

trap 'rc=$?; if [ "$rc" -eq 0 ]; then merge_compile_commands || true; fi; exit "$rc"' EXIT
build_nccl

if [ $ENABLE_NVSHMEM == "ON" ]; then
    if [ -n "$NVSHMEM_HOME" ]; then
        echo "Found NVSHMEM_HOME from environment variable: $NVSHMEM_HOME. skip install..."
    else
        echo "NVSHMEM_HOME is not set, try using NVSHMEM from pip..."
        # if not installed, install it from pip
        if [ -z "$(pip3 list | grep nvidia-nvshmem-cu12)" ]; then
            pip3 install nvidia-nvshmem-cu12==3.3.9
        fi
        NVSHMEM_HOME=$(python3 -c "import nvidia.nvshmem, pathlib; print(pathlib.Path(nvidia.nvshmem.__path__[0]))" 2>/dev/null)
        pushd $NVSHMEM_HOME/lib
        if [ ! -f libnvshmem_host.so ]; then
            ln -s libnvshmem_host.so.3 libnvshmem_host.so
        fi
        popd
    fi
fi

build_protobuf
build_flux_cuda
build_flux_py
