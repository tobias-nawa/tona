#!/usr/bin/env bash
# Baseline measurement: start a server, measure a suite, stop the server.
# Usage: scripts/baseline.sh <llamacpp-hip|llamacpp-vulkan> [arguments for bench/run run ...]
# Example: scripts/baseline.sh llamacpp-hip -c short-code,needle-10k -n 3 -l first-test
# Environment: MOE_CACHE_MIB (llama.cpp, default 8192), CTX (default 65536), FORCE=1 (RAM guard off)
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${1:?target missing: llamacpp-hip or llamacpp-vulkan}
shift
MODEL=work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
CTX=${CTX:-65536}
MOE_CACHE_MIB=${MOE_CACHE_MIB:-8192}
mkdir -p work/logs
LOG=work/logs/$(date +%Y%m%d-%H%M%S)-$TARGET.log

# With 32 GB the reference server ran into the OOM killer twice
mem_gib=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
if [ "$mem_gib" -lt 60 ] && [ "${FORCE:-0}" != 1 ]; then
    echo "ERROR: only ${mem_gib} GiB RAM. Use the 64 GB kit first, or FORCE=1." >&2
    exit 1
fi
[ -e "$MODEL" ] || { echo "Model missing: tools/get_model.py coder-iq1_m --from ..." >&2; exit 1; }

# shellcheck disable=SC1091
source toolchain/env.sh
case $TARGET in
    llamacpp-hip|llamacpp-vulkan)
        [ "$TARGET" = llamacpp-hip ] && PORT=8081 || PORT=8082
        BIN=work/build/llama.cpp-${TARGET#llamacpp-}/bin/llama-server
        [ -x "$BIN" ] || { echo "$BIN missing: scripts/build-llamacpp.sh" >&2; exit 1; }
        # Physical cores, as Tona's own default (backend/runtime_env.h); 6 was the Ryzen 5 9600X
        THREADS=${LLAMA_THREADS:-$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)}
        # All experts in RAM, GPU cache for hot experts, int8 KV, 64K context. No MTP: the GGUF has no nextn layers.
        cmd=("$BIN" -m "$MODEL" --port "$PORT" -c "$CTX" -np 1 -ngl 999 --cpu-moe --moe-cache-mib "$MOE_CACHE_MIB"
             -fa on -ctk q8_0 -ctv q8_0 -t "$THREADS" --jinja --metrics) ;;
    *) echo "unknown target $TARGET" >&2; exit 1 ;;
esac

if curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    echo "ERROR: port $PORT is already in use" >&2
    exit 1
fi
echo "Starting $TARGET, log: $LOG" >&2
printf '%q ' "${cmd[@]}" > "$LOG"; echo >> "$LOG"
"${cmd[@]}" >> "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null || true' EXIT

# Loading takes a while: 58 GB from NVMe, then filling the expert cache
for i in $(seq 1 900); do
    if curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
        echo "ready after ${i} s" >&2
        break
    fi
    kill -0 $PID 2>/dev/null || { echo "Server exited, see $LOG" >&2; tail -20 "$LOG" >&2; exit 1; }
    sleep 1
done
bench/run run -t "$TARGET" "$@"
cp "$LOG" "work/logs/last-$TARGET.log"
