#!/bin/bash
# with_server.sh LOG CTX CMD...: start the engine server on port 8092 under a memory limit, run CMD (PORT is set in its
# environment), stop the server. Hold the GPU lock around the whole call: flock $TONA_GPU_LOCK with_server.sh ...
# Environment: BUILD (default engine, a directory under work/build), DRAFT (mtp or dflash, default mtp), TONA_* (passed on),
# W76_MODEL_DIR (default work/models/coder-iq1_m), W76_PROFILE (expert profile, default work/profile/experts-coder.bin),
# W76_PORT (default 8092).
set -u
log=$1 ctx=$2
shift 2
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root" || exit 1
build=${BUILD:-engine}
draft=${DRAFT:-mtp}
port=${W76_PORT:-8092}
model=${W76_MODEL_DIR:-work/models/coder-iq1_m}/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
case $draft in
mtp) draft_args=(--mtp work/models/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf) ;;
dflash) draft_args=(--dflash work/models/draft/dflash-Qwen3.8-Flash-Next-Q8_0.gguf) ;;
*) echo "DRAFT must be mtp or dflash" >&2; exit 2 ;;
esac
# Leave 6 GB to the desktop, as the other measurement scripts do; no swap, so a runaway process dies instead of freezing.
limit=$(($(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1048576 - 6))G
systemd-run --user --scope -q -p "MemoryMax=$limit" -p MemorySwapMax=0 env OMP_NUM_THREADS=6 \
  "TONA_EXPERT_PROFILE=$PWD/${W76_PROFILE:-work/profile/experts-coder.bin}" \
  "work/build/$build/tona-server" "$model" "${draft_args[@]}" --port "$port" --ctx "$ctx" >"$log" 2>&1 &
spid=$!
for _ in $(seq 1 300); do
  kill -0 "$spid" 2>/dev/null || { echo "server exited, see $log" >&2; exit 1; }
  curl -fs http://127.0.0.1:"$port"/status 2>/dev/null | python3 -I -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("ready") else 1)' 2>/dev/null && break
  sleep 2
done
PORT=$port "$@"
rc=$?
pid=$(ss -ltnp 2>/dev/null | grep ":$port " | grep -o 'pid=[0-9]*' | cut -d= -f2)
[ -n "$pid" ] && kill "$pid"
while [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; do sleep 1; done
grep -E "barrier aborted" "$log" || true
exit $rc
