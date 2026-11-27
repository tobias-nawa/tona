#!/usr/bin/env bash
# Golden output check for refactors: greedy text of fixed prompts in deterministic mode, compared byte for byte.
# Usage: scripts/golden.sh record|compare [build dir, default work/build/engine]
# The prompts of a variant run in one tona-gpu-gen process (--prompts-file, state reset between them).
# record writes work/golden/<prompt>.txt; compare runs again and diffs against them. Run with the GPU otherwise idle.
# GOLDEN_PLAN=gpu|cpu (default gpu) is passed as TONA_PLAN. The gpu references live in work/golden/, the cpu
# references (experts on the CPU) in work/golden/cpu/.
# GOLDEN_MODEL=mini runs the mini model (tools/make_mini_model.py) without a drafter and without the expert
# profile; references in work/golden/mini/ (the same sub-folders). It checks host logic, not the kernels' numbers on
# real weights.
# GOLDEN_MODE=prefill runs the ~13K-token prompt scripts/bench-long-code.txt with --prefill --ctx 16384 (the default
# context is too small), then decodes 64 tokens. tona-gpu-gen takes the prefill path only with a drafter, so only
# GOLDEN_DRAFTER=dflash (references in prefill-dflash/) checks the prefill kernels; without it the prompt runs through
# the megakernel in verify passes of width 4 (--prompt-nb 4, 157 s instead of 359 s for single steps) and the 64
# decoded tokens run at width 1 on that state. What this no longer checks: the width-1 kernel path over the 13K prompt
# positions. Width 4 and width 1 are bit-identical (tona-gpu-gen --batch-check 13328: 13328 of 13328 positions equal,
# research/measurements/2026-11-02-golden-prefill-nb.md), so the reference is the one recorded with single steps.
# GOLDEN_PROMPT_NB=1 restores single steps (the full check; run it in the scale run and before a release).
# References: work/golden/prefill/ and work/golden/prefill-dflash/ (under work/golden/cpu/ for the cpu plan).
set -euo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/gpu-lock.sh"
cd "$(dirname "$0")/.."
mode=${1:?record or compare}
build=${2:-work/build/engine}
# A stale build proves nothing: exit 2 unless the binary is newer than its sources
scripts/build-fresh.sh "$build" tona-gpu-gen || exit 2
# GOLDEN_MODEL_FILE=path: another file of the same weights (a repacked variant must give the references of the original)
model=${GOLDEN_MODEL_FILE:-work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf}
drafter=work/models/draft/dflash-Qwen3.8-Flash-Next-Q8_0.gguf
plan=${GOLDEN_PLAN:-gpu}
variant=${GOLDEN_MODE:-decode}
ref=work/golden
profile=$PWD/work/profile/experts-coder.bin
if [[ ${GOLDEN_MODEL:-big} == mini ]]; then
    model=work/models/mini/mini.gguf
    ref=work/golden/mini
    profile=""
    drafter=""
    [[ ${GOLDEN_DRAFTER:-none} != dflash ]] || { echo "GOLDEN_MODEL=mini has no dflash drafter"; exit 2; }
fi
[[ $plan == gpu ]] || ref=$ref/$plan
[[ $variant == decode ]] || ref=$ref/$variant
# GOLDEN_DRAFTER=dflash (prefill mode only) adds the drafter: prefill path plus its feature injection.
drafter_mode=${GOLDEN_DRAFTER:-none}
[[ $drafter_mode == dflash && $variant == prefill ]] && ref=$ref-dflash
mkdir -p work/tmp
out=$(mktemp -d work/tmp/golden.XXXXXX)
limit=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1048576 - 6 ))G

chat() { printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' "$1"; }
declare -A prompts=(
    [csv]="Write a Python function that parses a CSV file and returns the average of each numeric column. Only code."
    [rust]="Write a Rust function that checks whether a string is a valid IPv4 address, with tests. Only code."
    [explain]="Explain how a hash map handles collisions, in a few paragraphs."
)
names=(csv rust explain)
if [[ $variant == prefill ]]; then
    names=(long)
    prompts[long]=$(cat scripts/bench-long-code.txt)
fi
# One process for all prompts of the variant: the model loads once (18 s) and tona-gpu-gen resets the state between the
# prompts (--prompts-file). The references are the ones of one process per prompt, byte for byte.
list=$out/prompts.list
for name in "${names[@]}"; do
    if [[ $variant == prefill ]]; then text=${prompts[$name]}; else text=$(chat "${prompts[$name]}"); fi
    printf '%s' "$text" >"$out/$name.prompt"
    printf '%s\t%s\n' "$out/$name.txt" "$out/$name.prompt" >>"$list"
done
if [[ $variant == prefill ]]; then
    flags=(-n 64 --mega --prefill --ctx 16384)
    if [[ $drafter_mode == dflash ]]; then flags+=(--dflash "$drafter"); else flags+=(--prompt-nb "${GOLDEN_PROMPT_NB:-4}"); fi
else
    flags=(-n 200 --mega ${drafter:+--dflash "$drafter"})
fi
status=0
systemd-run --user --scope -q -p MemoryMax="$limit" -p MemorySwapMax=0 \
    env TONA_DETERMINISTIC=1 TONA_PLAN="$plan" ${profile:+TONA_EXPERT_PROFILE="$profile"} \
    "$build/tona-gpu-gen" "$model" --prompts-file "$list" "${flags[@]}" \
    >/dev/null 2>"$out/gen.log" || { echo "run failed, see $out/gen.log"; exit 1; }
for name in "${names[@]}"; do
    if [[ $mode == record ]]; then
        mkdir -p "$ref" && cp "$out/$name.txt" "$ref/$name.txt" && echo "$name: recorded"
    elif cmp -s "$ref/$name.txt" "$out/$name.txt"; then
        echo "$name: identical"
    else
        echo "$name: DIFFERS (diff $ref/$name.txt $out/$name.txt)"; status=1
    fi
done
exit $status
