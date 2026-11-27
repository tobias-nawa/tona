#!/bin/bash
# nll.sh TEXT PREFILL TAIL TAG [VAR=value ...]: prefill PREFILL tokens of TEXT, then TAIL tokens teacher-forced through the
# megakernel (the decode path; method: research/measurements/2026-10-29-indexcache.md). TAG=ref writes the logits dump
# (TONA_TEACHER_DUMP, 1 MB per token); any other tag compares against the dump of the same text, prefill and tail
# (TONA_TEACHER_REF) and writes one line per token to nll/<name>-<tag>.txt (format of scripts/nll-compare.py). Hold the
# GPU lock around the call.
set -u
text=$1 pre=$2 tail=$3 tag=$4
shift 4
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root" || exit 1
# shellcheck disable=SC1091
source toolchain/env.sh
w=${W76_WORK:-$root/work/tmp/w76}/nll
name=$(basename "$text" .txt)-$pre-$tail
if [ "$tag" = ref ]; then x="TONA_TEACHER_DUMP=$w/$name.bin"; else x="TONA_TEACHER_REF=$w/$name.bin TONA_TEACHER_OUT=$w/$name-$tag.txt"; fi
limit=$(($(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1048576 - 6))G
# $x is two assignments at most; word splitting is intended
# shellcheck disable=SC2086
systemd-run --user --scope -q -p "MemoryMax=$limit" -p MemorySwapMax=0 env OMP_NUM_THREADS=6 "TONA_TEACHER_TAIL=$tail" $x "$@" \
  "work/build/${BUILD:-w76}/tona-gpu-gen" work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf "" \
  --mega --teacher "$text" --max $((pre + tail)) --ctx $((pre + tail + 1024)) 2>&1 | grep -E "^tail:|rror|abort" | sed "s/^/[$name $tag $*] /"
