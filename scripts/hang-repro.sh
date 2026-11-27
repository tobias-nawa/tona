#!/usr/bin/env bash
# Reproduce the GPU hang of 23 Oct: tona-server with 32K context under the watchdog (scripts/gpu-watch.sh),
# then a smoke test and the recorded Claude Code request (work/hang/replay/*.json, ~7.4K tokens).
# Usage: scripts/hang-repro.sh [rounds, default 3] [VAR=VALUE ...]  e.g. TONA_VRAM_RESERVE_MB=2000 TONA_MEGA_PROF=1
# Result: work/hang/<time>/ (watch.log, kernel.log, on a hang umr-*.txt and rocgdb.txt) and server.log
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck disable=SC1091
source toolchain/env.sh
rounds=${1:-3} port=8096 hb=/run/user/$(id -u)/tona-heartbeat
shift || true
limit=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1048576 - 6 ))G  # as in the justfile
model=work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
compgen -G "work/hang/replay/*.json" >/dev/null || { echo "no recordings in work/hang/replay/"; exit 1; }
rm -f "$hb"

echo "Environment: $* (memory limit $limit)"
systemd-run --user --scope -q -p MemoryMax="$limit" -p MemorySwapMax=0 \
    env "$@" TONA_HEARTBEAT="$hb" TONA_EXPERT_PROFILE="$PWD/work/profile/experts-coder.bin" \
    work/build/engine/tona-server "$model" --dflash work/models/draft/dflash-Qwen3.8-Flash-Next-Q8_0.gguf \
    --port "$port" --ctx 32768 > work/tmp/hang-server.log 2>&1 &
until spid=$(pgrep -nx tona-server); do sleep 0.2; done
scripts/gpu-watch.sh "$spid" "$hb" 15 &
wpid=$!
until curl -sf localhost:$port/health >/dev/null; do
    kill -0 "$spid" 2>/dev/null || { echo "Server dead"; cat work/tmp/hang-server.log; exit 1; }
    sleep 1
done

for r in $(seq "$rounds"); do
    echo "== Round $r: smoke test"
    timeout 600 /usr/bin/python3 -I tools/server_smoke.py http://127.0.0.1:$port | tail -1
    for f in work/hang/replay/*.json; do
        echo "== Round $r: $(basename "$f")"
        # answer limited to 600 tokens, otherwise as sent by Claude Code
        /usr/bin/python3 -I -c 'import json,sys; r=json.load(open(sys.argv[1])); r["max_tokens"]=600; print(json.dumps(r))' "$f" |
            timeout 900 curl -sN localhost:$port/v1/messages -H 'content-type: application/json' -d @- |
            grep -c '^event: content_block_delta' | sed 's/^/   deltas: /'
        kill -0 "$spid" 2>/dev/null || break 2
    done
done

kill "$spid" 2>/dev/null
wait "$wpid"
d=$(find work/hang -maxdepth 1 -name '2*' | sort | tail -1)
cp work/tmp/hang-server.log "$d/server.log"
# Log words of tona-server this grep relies on: adapt it when the server log format changes
grep -E "Request|Error" "$d/server.log" | tail -20
echo "Log: $d"
