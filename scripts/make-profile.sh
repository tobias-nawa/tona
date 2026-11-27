#!/usr/bin/env bash
# Create an expert profile: how often each layer picks which expert, over coding prompts and one Claude Code
# request (tool use). At start the engine puts the most frequent experts into VRAM and pins the next ones in RAM.
# Regenerate per model version: the expert numbers only hold for one GGUF.
# Usage: scripts/make-profile.sh MODEL.gguf OUTPUT.bin [tokens per prompt, default 250]
# The measurement prompts of bench-decode.sh are deliberately not included.
set -euo pipefail
cd "$(dirname "$0")/.."
M=$1 OUT=$2 N=${3:-250}
# shellcheck disable=SC1091
source toolchain/env.sh
LIMIT=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1048576 - 6 ))G
chat() { printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' "$1"; }
prompts=(
    "$(chat 'Write a Python class for an LRU cache with get, put and a capacity limit, with unit tests.')"
    "$(chat 'Implement a TypeScript function that debounces another function, with types and examples.')"
    "$(chat 'Write a Go HTTP handler that accepts JSON, validates it and stores it in PostgreSQL.')"
    "$(chat 'Refactor this C++ loop to use std::ranges and explain the change: for (int i = 0; i < v.size(); i++) if (v[i] % 2) out.push_back(v[i] * 2);')"
    "$(chat 'Write a bash script that backs up a directory with rsync and rotates the last 7 backups.')"
    "$(chat 'Explain what a race condition is and fix the one in this Rust snippet: static mut COUNTER: u32 = 0; fn inc() { unsafe { COUNTER += 1; } }')"
)
if [ -f work/hang/replay/req1-_v1_messages.json ]; then
    prompts+=("$(work/build/engine/tona-chat --anthropic work/hang/replay/req1-_v1_messages.json 2>/dev/null)")
fi
rm -f "$OUT"
for i in "${!prompts[@]}"; do
    # the profile accumulates: counts read at start plus new ones are written on exit
    pre=()
    [ -f "$OUT" ] && pre=(TONA_EXPERT_PROFILE="$PWD/$OUT")
    systemd-run --user --scope -q -p MemoryMax="$LIMIT" -p MemorySwapMax=0 \
        env "${pre[@]}" TONA_EXPERT_PROFILE_OUT="$PWD/$OUT" \
        work/build/engine/tona-gpu-gen "$M" "${prompts[$i]}" -n "$N" --ctx 16384 --mega 2>&1 | grep -E '^Error' || true
    echo "Prompt $((i + 1)) of ${#prompts[@]}"
done
ls -la "$OUT"
