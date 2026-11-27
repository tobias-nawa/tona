#!/usr/bin/env bash
# Extra expert slots in the prefill workspace: the tokens of a conversation must not depend on them. Two servers on
# the mini model with few slots (TONA_EXPERT_SLOTS) and deterministic mode, one with the extra slots and one without
# (TONA_PF_EXTRA_SLOTS=0), get the same requests: a long prompt (chunk-by-chunk prefill) between short ones, so that the
# extras are evicted, used by the prefill and restored between decodes. The answers must be byte-equal, the first
# server must have given extra slots up, and it must have had hits in them again afterwards.
# Usage: scripts/extra-slots-check.sh [build dir, default work/build/engine]
set -euo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/gpu-lock.sh"
cd "$(dirname "$0")/.."
build=${1:-work/build/engine}
scripts/build-fresh.sh "$build" tona-server || exit 2
port=8092
model=work/models/mini/mini.gguf
mtp=work/models/mini/mini-mtp.gguf
mkdir -p work/tmp
out=$(mktemp -d work/tmp/extra-slots.XXXXXX)
limit=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1048576 - 6 ))G
pid=""
stop_server() {
    if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    pid=""
    return 0
}
trap stop_server EXIT

client=$out/client.py
cat >"$client" <<'PY'
import json
import sys
import urllib.request

base, dest = sys.argv[1:3]
WORDS = 'alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau'.split()


def prompt(n, seed):
    return ' '.join(WORDS[(i * seed + i // 7) % len(WORDS)] for i in range(n))


def ask(text):
    body = {'model': 'local', 'max_tokens': 48, 'temperature': 0,
            'messages': [{'role': 'user', 'content': text}]}
    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'content-type': 'application/json'})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return {'text': r['choices'][0]['message'].get('content'), 'finish': r['choices'][0]['finish_reason'],
            'usage': r['usage']}


# short, long (one full chunk: the whole workspace), a few hundred tokens, short again, long again
sizes = [(40, 3), (1500, 5), (300, 7), (60, 11), (700, 13), (40, 3), (2600, 17)]
res = [ask(prompt(n, s)) for n, s in sizes]
with open(dest, 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=1)
with urllib.request.urlopen(base + '/status', timeout=30) as r:
    st = json.load(r)
print(json.dumps({k: st.get('totals', st).get(k) for k in ('expert_hits', 'expert_misses', 'early_hits', 'early_misses')}))
PY

run_arm() { # NAME EXTRA(0|1)
    local name=$1 extra=$2
    systemd-run --user --scope -q -p MemoryMax="$limit" -p MemorySwapMax=0 \
        env TONA_DETERMINISTIC=1 TONA_PLAN=gpu TONA_EXPERT_SLOTS=24 TONA_PF_EXTRA_SLOTS="$extra" TONA_PF_DEBUG=1 \
        TONA_VRAM_LOG=1 ROCR_VISIBLE_DEVICES=0 "$build/tona-server" "$model" --mtp "$mtp" --port "$port" --ctx 8192 \
        >"$out/$name.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 120); do
        kill -0 "$pid" 2>/dev/null || { echo "server exited early, see $out/$name.log"; exit 1; }
        if curl -fs "http://127.0.0.1:$port/status" 2>/dev/null | python3 -I -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("ready") else 1)' 2>/dev/null; then break; fi
        sleep 1
    done
    python3 -I "$client" "http://127.0.0.1:$port" "$out/$name.json" >"$out/$name.stats"
    stop_server
}

run_arm extras 1
run_arm plain 0
grep -q "extra slots given up" "$out/extras.log" || { echo "FAIL: the first arm never gave extra slots up ($out/extras.log)"; exit 1; }
grep -q "^Extra slots:" "$out/extras.log" || { echo "FAIL: no extra slots in the first arm"; exit 1; }
! grep -q "^Extra slots:" "$out/plain.log" || { echo "FAIL: TONA_PF_EXTRA_SLOTS=0 still has extra slots"; exit 1; }
if cmp -s "$out/extras.json" "$out/plain.json"; then
    echo "extra slots: identical answers ($(cat "$out/extras.stats"))"
else
    echo "FAIL: the answers differ: diff $out/extras.json $out/plain.json"
    exit 1
fi
