# shellcheck shell=bash
# Activate the engine toolchain: source toolchain/env.sh
# Puts ROCm 10.1 from work/rocm-10.1 (scripts/setup.sh) ahead of everything from the system.

_tona_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
_tona_venv="$_tona_root/work/rocm-10.1"
if [ ! -x "$_tona_venv/bin/hipconfig" ]; then
    echo "toolchain/env.sh: $_tona_venv is missing, run scripts/setup.sh first" >&2
    # shellcheck disable=SC2317  # reachable: the file is sourced (return) or executed (exit)
    return 1 2>/dev/null || exit 1
fi

export VIRTUAL_ENV="$_tona_venv"
export PATH="$_tona_venv/bin:$PATH"
ROCM_PATH="$("$_tona_venv/bin/hipconfig" --rocmpath)"
export ROCM_PATH
export HIP_PATH="$ROCM_PATH"
# Only the discrete GPU (gfx1200), not the iGPU (gfx1036)
export ROCR_VISIBLE_DEVICES=0
export GGML_VK_VISIBLE_DEVICES=0   # llama.cpp Vulkan: Vulkan1 would be the iGPU
export TONA_GPU_ARCH=gfx1200
# Compiler directly, without hipcc. Headers from ROCM_PATH explicitly first: otherwise clang falls back to
# Fedora's /usr/include/hip (HIP 7.1) while linking against 7.16.
export TONA_HIPCXX="$ROCM_PATH/lib/llvm/bin/amdclang++"
export TONA_HIPFLAGS="-x hip --offload-arch=$TONA_GPU_ARCH -isystem $ROCM_PATH/include"
export TONA_HIPLDFLAGS="-L$ROCM_PATH/lib -lamdhip64 -Wl,-rpath,$ROCM_PATH/lib"

# TONA_DIAG=1 (a developer's machine): every GPU process leaves the allocation trace (one file per PID) and, at a
# fault, a GPU dump, so a rare fault is captured whichever run hits it. A run may set its own paths.
if [ -n "${TONA_DIAG:-}" ] && [ "${TONA_DIAG}" != 0 ]; then
    TONA_DIAG_DIR="${TONA_DIAG_DIR:-$_tona_root/work/diag}"
    export TONA_DIAG_DIR
    mkdir -p "$TONA_DIAG_DIR"
    export TONA_ALLOC_TRACE="${TONA_ALLOC_TRACE:-$TONA_DIAG_DIR/alloc-trace-%p.txt}"
    export TONA_KFD_SMI_LOG="${TONA_KFD_SMI_LOG:-$TONA_DIAG_DIR/kfd-smi-%p.log}"
    export HSA_COREDUMP_PATTERN="${HSA_COREDUMP_PATTERN:-$TONA_DIAG_DIR/gpucore-%p.core}"
    # RLIMIT_CORE also limits the GPU dump
    # (bash may clamp to the hard limit without an error, so the result is checked)
    ulimit -S -c unlimited 2>/dev/null || true
    [ "$(ulimit -S -c)" = unlimited ] || echo "toolchain/env.sh: core limit is $(ulimit -S -c), no GPU dump" >&2
    "$_tona_root/scripts/diag-clean.sh" "$TONA_DIAG_DIR"
fi
unset _tona_root _tona_venv
