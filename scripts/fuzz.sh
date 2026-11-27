#!/bin/bash
# Fuzzing of the server inputs (API parsing, chat rendering, output parser), the GGUF loader and the update signature and manifest parsers with libFuzzer, ASan and UBSan.
# Usage: scripts/fuzz.sh [seconds per target, default 300] [target...]   (targets: api gguf minisig; default all)
# Every target has its own folder $FUZZ_OUT/<target>/ (default work/fuzz) with the fuzzer, its corpus and its findings,
# so two runs of different targets never share a file. The compiler is the ROCm toolchain's clang++, or FUZZ_CXX
# (CI: the clang of the distribution). All targets run; the exit code is 1 when any of them found a crash, a leak, a
# timeout or an out-of-memory input, and the inputs are in $FUZZ_OUT/<target>/findings/.
set -euo pipefail
cd "$(dirname "$0")/.."
secs=${1:-300}
[ $# -eq 0 ] || shift
CXX=${FUZZ_CXX:-$(find work/rocm-10.1/lib -path '*/site-packages/_rocm_sdk_core/lib/llvm/bin/clang++' | sort | head -1)}
[ -n "$CXX" ] || { echo "fuzz: no clang++ (set FUZZ_CXX, or run scripts/setup.sh)" >&2; exit 2; }
out=${FUZZ_OUT:-work/fuzz}
wanted=("$@")
status=0

# fuzz TARGET SOURCE...: builds engine/tests/fuzz/fuzz_<target>.cpp with the given engine sources and runs it
fuzz() {
    local target=$1
    shift
    if [ ${#wanted[@]} -gt 0 ] && [[ ! " ${wanted[*]} " == *" $target "* ]]; then return 0; fi
    local dir=$out/$target
    mkdir -p "$dir/corpus" "$dir/findings"
    "$CXX" -std=c++23 -O1 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined -Iengine \
        "engine/tests/fuzz/fuzz_$target.cpp" "$@" -o "$dir/fuzz"
    cp -n "engine/tests/fuzz/seeds/$target"/* "$dir/corpus/" 2>/dev/null || true
    "$dir/fuzz" -max_total_time="$secs" -max_len=8192 -timeout=10 -rss_limit_mb=4096 \
        -artifact_prefix="$dir/findings/" -print_final_stats=1 "$dir/corpus" || status=1
}

fuzz api engine/server/{api,api_anthropic,api_openai,chat,chat_format,chat_qwen3_render,chat_qwen3_parser,chat_nemotron_render}.cpp
fuzz gguf engine/model/{gguf,quant,quant_tile}.cpp
fuzz minisig engine/update/{crypto,minisig,manifest,url}.cpp -lcrypto
exit "$status"
