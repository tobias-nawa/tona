#!/usr/bin/env bash
# Builds tools/llama_logits_dump.cpp against the pinned llama.cpp (work/src/llama.cpp, build in work/build/llama.cpp-hip)
# into work/build/llama.cpp-hip/bin/llama-logits-dump. Nothing of the engine build depends on it.
set -euo pipefail
cd "$(dirname "$0")/.."
src=work/src/llama.cpp
bin=work/build/llama.cpp-hip/bin
[[ -f $bin/libllama.so ]] || { echo "build llama.cpp first ($bin/libllama.so missing)"; exit 1; }
c++ -std=c++17 -O2 tools/llama_logits_dump.cpp -I"$src/include" -I"$src/ggml/include" -L"$bin" \
    -lllama -lggml -lggml-base -Wl,-rpath,"$(pwd)/$bin" -o "$bin/llama-logits-dump"
echo "$bin/llama-logits-dump"
