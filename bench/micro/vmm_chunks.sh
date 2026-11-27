#!/usr/bin/env bash
# E5 (docs/architecture/vram-giveback.md): count the KFD allocation ioctls of vmm_chunks with 0 and N chunks.
# Usage (inside the GPU queue, after `source toolchain/env.sh`): bench/micro/vmm_chunks.sh OUTDIR [CHUNKS, default 260]
set -euo pipefail
out=${1:?output directory}
n=${2:-260}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
# shellcheck disable=SC2086  # the flag variables hold several words
"$TONA_HIPCXX" $TONA_HIPFLAGS -O2 -std=c++20 "$here/vmm_chunks.hip" -o "$out/vmm_chunks" $TONA_HIPLDFLAGS
for k in 0 "$n"; do
    strace -f -e trace=ioctl -o "$out/strace-$k.txt" "$out/vmm_chunks" "$k" 32 | tee "$out/run-$k.txt"
done
# KFD ioctl numbers (include/uapi/linux/kfd_ioctl.h, type 'K' = 0x4b): ALLOC_MEMORY_OF_GPU 0x16, MAP_MEMORY_TO_GPU 0x18,
# FREE_MEMORY_OF_GPU 0x17
count() { grep -cE "0x4b, $2[,)]|$3" "$out/strace-$1.txt" || true; }
for nr in 0x16:alloc:ALLOC_MEMORY_OF_GPU 0x18:map:MAP_MEMORY_TO_GPU 0x17:free:FREE_MEMORY_OF_GPU; do
    IFS=: read -r num name sym <<<"$nr"
    echo "ioctl $name ($num): baseline $(count 0 "$num" "$sym"), with $n chunks $(count "$n" "$num" "$sym")"
done | tee "$out/ioctl-counts.txt"
