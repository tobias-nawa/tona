#!/usr/bin/env bash
# Determinism check of the Bolt (or Ornith) prefill and decode: the same prompt must give the same logits and the same
# tokens under TONA_DETERMINISTIC=1 in every prefill plan and in every repeat (the expert cache state carries over from one
# repeat to the next). Plans: the default one, chunk by chunk (TONA_PF_LAYERED=0), layer by layer at two loan sizes
# (TONA_PF_LOAN_PCT 75, 90) and a layered chunk of 1024. One line per run; exit 1 if the hashes of the first decode
# logits or of the generated ids are not all equal. The second column is the plan that really ran: a layered plan needs
# a loan limit that fits (more free VRAM, 122 slots per layer or more for Bolt), so on a busy GPU some stay chunk by chunk;
# check that at least one run says "layer by layer" before trusting a green result (TONA_PF_LOAN_PCT=90 usually does).
# Usage: scripts/bolt-determinism.sh [build dir, default work/build/engine]
# Environment: BOLT_MODEL (GGUF, default the Bolt file under work/models), BOLT_IDS (token ids, one per line or
# blank separated; default: the prose corpus tokenized once into work/corpus/ids-bolt-prose.txt), BOLT_PROMPT (tokens,
# default 6000: three chunks of 2048, so the layered plans have several), BOLT_PROFILE (expert profile).
# Why: a layered plan once computed another network for Bolt (the norm of a layer's input was taken from the wrong
# chunk, KL 11.5) and the free VRAM of a start decided the plan, so equal runs gave different tokens
# (research/measurements/2026-11-20-bolt-determinism.md).
set -euo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/gpu-lock.sh"
cd "$(dirname "$0")/.."
build=${1:-work/build/engine}
model=${BOLT_MODEL:-work/models/nemotron-bolt-74b-a9b-24gb/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf}
profile=${BOLT_PROFILE:-toolchain/profiles/nemotron-bolt-74b-a9b-24gb.bin}
prompt=${BOLT_PROMPT:-6000}
ids=${BOLT_IDS:-work/corpus/ids-bolt-prose.txt}
scripts/build-fresh.sh "$build" tona-bolt-bench || exit 2
if [[ ! -s $ids ]]; then
    [[ -s work/corpus/prose.txt ]] || { echo "work/corpus/prose.txt is missing (scripts/make_prose_corpus.py)"; exit 2; }
    "$build/tona-tokenize" "$model" work/corpus/prose.txt --no-special >"$ids"
fi

plans=("default:" "chunked:TONA_PF_LAYERED=0" "layered75:TONA_PF_LOAN_PCT=75" "layered90:TONA_PF_LOAN_PCT=90"
    "layered-c1024:TONA_PREFILL_CHUNK=1024")
fail=0
ref_logits="" ref_ids=""
for plan in "${plans[@]}"; do
    name=${plan%%:*}
    vars=${plan#*:}
    # shellcheck disable=SC2086  # the variable list is split on purpose
    out=$(env TONA_DETERMINISTIC=1 TONA_EXPERT_PROFILE="$profile" TONA_PF_DEBUG=1 $vars \
        "$build/tona-bolt-bench" "$model" "$ids" --prompt "$prompt" --gen 64 --ctx 262144 --repeat 2 2>&1)
    kind=$(grep -m1 -o 'layer by layer\|chunk by chunk' <<<"$out" || true)
    while read -r logits ids_hash; do
        [[ -n $ref_logits ]] || ref_logits=$logits ref_ids=$ids_hash
        verdict=ok
        if [[ $logits != "$ref_logits" || $ids_hash != "$ref_ids" ]]; then verdict=DIFFERENT fail=1; fi
        printf '%-14s %-14s logits %s ids %s %s\n' "$name" "${kind:-?}" "$logits" "$ids_hash" "$verdict"
    done < <(paste -d' ' <(grep -o 'first logits hash: [0-9a-f]*' <<<"$out" | cut -d' ' -f4) \
        <(grep -o 'generated ids hash: [0-9a-f]*' <<<"$out" | cut -d' ' -f4))
done
exit $fail
