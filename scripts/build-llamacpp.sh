#!/usr/bin/env bash
# Builds llama.cpp as a yardstick: pinned tag, ROCm 10.1 from toolchain/env.sh, and Vulkan.
# Result: work/build/llama.cpp-{hip,vulkan}/bin/{llama-server,llama-bench}
# Usage: scripts/build-llamacpp.sh [hip|vulkan|all]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=b11822
SHA=d9ee007a06b63e9b2380ba28663de4bcc7b60260   # commit of the tag, checked after cloning
SRC=work/src/llama.cpp
WHAT=${1:-all}
TARGETS=(llama-server llama-bench llama-cli)

if [ ! -d "$SRC/.git" ]; then
    git clone --quiet --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp.git "$SRC"
fi
have=$(git -C "$SRC" rev-parse HEAD)
[ "$have" = "$SHA" ] || { echo "ERROR: $SRC is at $have, expected $SHA ($TAG)" >&2; exit 1; }

if [ "$WHAT" = hip ] || [ "$WHAT" = all ]; then
    # shellcheck disable=SC1091
    source toolchain/env.sh
    B=work/build/llama.cpp-hip
    # Headers from ROCM_PATH explicitly first, see toolchain/env.sh
    HIPCXX="$ROCM_PATH/lib/llvm/bin/clang" HIP_PATH="$ROCM_PATH" cmake -S "$SRC" -B "$B" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS="$TONA_GPU_ARCH" \
        -DCMAKE_PREFIX_PATH="$ROCM_PATH" -DCMAKE_HIP_FLAGS="-isystem $ROCM_PATH/include" \
        -DCMAKE_BUILD_RPATH="$ROCM_PATH/lib" -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_BUILD_NUMBER="${TAG#b}" \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
    cmake --build "$B" --target "${TARGETS[@]}" -j "$(nproc)"
fi

if [ "$WHAT" = vulkan ] || [ "$WHAT" = all ]; then
    B=work/build/llama.cpp-vulkan
    cmake -S "$SRC" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_NUMBER="${TAG#b}" \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    cmake --build "$B" --target "${TARGETS[@]}" -j "$(nproc)"
fi
echo "llama.cpp $TAG ($SHA) built: $WHAT"
