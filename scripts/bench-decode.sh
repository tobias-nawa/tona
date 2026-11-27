#!/usr/bin/env bash
# Decode measurement of the engine: three prompts (code, code, explanation), RUNS runs each, speculation with the
# DFlash2 drafter, everything automatic (slots from VRAM, pinned experts from RAM, profile if present).
# Usage: scripts/bench-decode.sh [runs, default 3] [more environment variables, e.g. TONA_PLAN=cpu]
# Environment: TONA_BUILD = build directory holding tona-gpu-gen (default work/build/engine), so before/after runs can
# compare two builds without copying the script.
# Output: per prompt the median and range of tok/s (decode), plus hits and the breakdown of misses.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=${1:-3}
shift || true
M=work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
D=work/models/draft/dflash-Qwen3.8-Flash-Next-Q8_0.gguf
BIN=${TONA_BUILD:-work/build/engine}/tona-gpu-gen
PROFILE=$PWD/work/profile/experts-coder.bin
OUT=$({ mkdir -p work/tmp; mktemp -d work/tmp/bench-decode.XXXXXX; })
# Memory limit: the process plus pinned experts; the rest of the RAM stays with the system
MEM_GB=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1048576 ))
LIMIT=$(( MEM_GB - 6 ))G

chat() { printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' "$1"; }
declare -A P
P[csv]=$(chat "Write a Python function that parses a CSV file and returns the average of each numeric column. Only code.")
P[rust]=$(chat "Write a Rust function that checks whether a string is a valid IPv4 address, with tests. Only code.")
P[explain]=$(chat "Explain how a hash map handles collisions, in a few paragraphs.")

echo "== $(date -I) $(git rev-parse --short HEAD), RAM ${MEM_GB} GB, limit ${LIMIT}, $RUNS runs, $*"
for name in csv rust explain; do
    values=()
    for i in $(seq 1 "$RUNS"); do
        log=$OUT/$name-$i.log
        systemd-run --user --scope -q -p MemoryMax="$LIMIT" -p MemorySwapMax=0 \
            env TONA_MEGA_INFO=1 TONA_EXPERT_PROFILE="$PROFILE" "$@" \
            "$BIN" "$M" "${P[$name]}" -n 200 --mega --dflash "$D" >"$OUT/$name-$i.txt" 2>"$log" || {
            echo "$name run $i: failed, see $log"; tail -3 "$log"; continue; }
        # log text of tona-gpu-gen: engine/tools/tona_gpu_gen_spec_report.cpp
        values+=("$(grep -oP 'per pass, \K[0-9.]+(?= tok/s)' "$log" | tail -1)")
    done
    last=$OUT/$name-$RUNS.log
    printf '%-8s tok/s: %s\n' "$name" "$(printf '%s\n' "${values[@]}" | sort -n | awk '{a[NR]=$1} END {if (NR) printf "median %.1f, range %.1f-%.1f (n = %d)", a[int((NR+1)/2)], a[1], a[NR], NR}')"
    grep -hE 'pinned|[Tt]okens per pass|Experts:|Misses per pass' "$last" | sed 's/^/      /'
done
echo "Logs: $OUT"
