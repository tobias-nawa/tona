#!/usr/bin/env bash
# Golden output check for the HTTP server: fixed requests against both APIs, greedy or seeded, deterministic mode,
# content only (text, tool calls, stop reason, token counts), compared byte for byte.
# Usage: scripts/golden-server.sh record|compare [build dir, default work/build/engine]
# Cases (groups, GOLDEN_CASES=base,sampled,... selects; default: all, with the dflash drafter: base only):
#   base      plain, streamed, tool call and streamed tool call on both APIs (8 requests)
#   sampled   temperature 0.7, top_p 0.8, seed 42, both APIs, plain and streamed
#   thinking  Anthropic request with thinking and a small budget (budget forcing)
#   reuse     two-turn conversation, then a long conversation, an unrelated one, and the first again (RAM stash)
#   ssd       checkpoints on the SSD: long conversation, server restart, same conversation plus a turn
#   hit       a hit on a checkpoint computes what a cold run of the same prompt computes: start of the last
#             message, fork, warm-up, and the same prompt again, each against a server with TONA_PROMPT_SLOTS=1 (no
#             checkpoints, so the prompt goes through the prefill in one piece); the pairs must be byte-equal
# GOLDEN_DRAFTER=mtp (default) or dflash; dflash records into work/golden/server-dflash.
# record writes work/golden/server/<case>.json and never overwrites a file that exists (GOLDEN_FORCE=1 does);
# compare runs again into a temp dir and diffs against them.
# GOLDEN_MODEL=mini uses the mini model and its MTP head (tools/make_mini_model.py; no expert profile,
# references in work/golden/mini/server, only the mtp drafter). The answers are noise, the checks cover the host
# logic: both APIs, streaming, stash and restore, the prompt cache on the SSD.
# The test server listens on port 8091 (GOLDEN_PORT=n changes it) and is stopped by its own PID. Run with the GPU otherwise idle.
set -euo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/gpu-lock.sh"
cd "$(dirname "$0")/.."
mode=${1:?record or compare}
build=${2:-work/build/engine}
[[ $mode == record || $mode == compare ]] || { echo "usage: $0 record|compare [build dir]"; exit 2; }
# A stale build proves nothing: exit 2 unless the binary is newer than its sources
scripts/build-fresh.sh "$build" tona-server || exit 2
port=${GOLDEN_PORT:-8091}
# GOLDEN_MODEL_FILE=path: another file of the same weights (a repacked variant must give the references of the original)
model=${GOLDEN_MODEL_FILE:-work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf}
case ${GOLDEN_DRAFTER:-mtp} in
    mtp)
        drafter_args=(--mtp work/models/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf)
        ref=work/golden/server
        cases=${GOLDEN_CASES:-base,sampled,thinking,reuse,ssd,hit}
        ;;
    dflash)
        drafter_args=(--dflash work/models/draft/dflash-Qwen3.8-Flash-Next-Q8_0.gguf)
        ref=work/golden/server-dflash
        cases=${GOLDEN_CASES:-base}
        ;;
    *) echo "GOLDEN_DRAFTER must be mtp or dflash"; exit 2 ;;
esac
profile=$PWD/work/profile/experts-coder.bin
if [[ ${GOLDEN_MODEL:-big} == mini ]]; then
    [[ ${GOLDEN_DRAFTER:-mtp} == mtp ]] || { echo "GOLDEN_MODEL=mini has only the mtp drafter"; exit 2; }
    model=work/models/mini/mini.gguf
    drafter_args=(--mtp work/models/mini/mini-mtp.gguf)
    ref=work/golden/mini/server
    profile=""
fi
mkdir -p work/tmp
out=$(mktemp -d work/tmp/golden-server.XXXXXX)
results=$out/results
mkdir -p "$results"
limit=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1048576 - 6 ))G
has_case() { [[ ",$cases," == *",$1,"* ]]; }

pid=""
status=0
stop_server() {
    if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    pid=""
    return 0
}
# A GPU page fault kills the server (2026-11-02: once in 19 runs, not reproduced since). The allocation trace of
# every server (TONA_ALLOC_TRACE) lets the faulting address of the log be assigned to a buffer afterwards.
export TONA_ALLOC_TRACE=${TONA_ALLOC_TRACE:-$out/alloc-%p.txt}
# shellcheck disable=SC2329  # called by the EXIT trap
explain_fault() {
    local addr t
    addr=$(grep -o 'faulting addr: 0x[0-9a-f]*' "$out/server.log" 2>/dev/null | head -1 | cut -d' ' -f3) || true
    [[ -n $addr ]] || return 0
    for t in "$out"/alloc-*.txt; do
        [[ -f $t ]] || continue
        echo "GPU page fault at $addr, allocation trace $t:"
        python3 -I tools/alloc_trace_lookup.py "$t" "$addr" || true
    done
}
trap 'explain_fault; stop_server' EXIT

# Starts the server on a fresh prompt cache (no TONA_PROMPT_CACHE unless given as an argument) and waits until ready.
start_server() {
    systemd-run --user --scope -q -p MemoryMax="$limit" -p MemorySwapMax=0 \
        env TONA_DETERMINISTIC=1 TONA_PLAN=gpu ROCR_VISIBLE_DEVICES=0 \
        ${profile:+TONA_EXPERT_PROFILE="$profile"} "$@" \
        "$build/tona-server" "$model" "${drafter_args[@]}" --port "$port" --ctx 32768 \
        >>"$out/server.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 900); do
        kill -0 "$pid" 2>/dev/null || { echo "server exited early, see $out/server.log"; exit 1; }
        if curl -fs "http://127.0.0.1:$port/status" 2>/dev/null | python3 -I -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("ready") else 1)' 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    echo "server not ready after 900 s, see $out/server.log"
    exit 1
}

# shellcheck source=/dev/null
source toolchain/env.sh
client=$out/client.py
cat >"$client" <<'PY'
import json
import sys
import urllib.request

base, dest, long_file = sys.argv[1:4]
groups = sys.argv[4:]
TOOL = {'name': 'read_file', 'description': 'Read a text file and return its content',
        'input_schema': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}
OPENAI_TOOL = {'type': 'function', 'function': {'name': TOOL['name'], 'description': TOOL['description'],
                                                'parameters': TOOL['input_schema']}}
PROMPT = 'Write a Python function that parses a CSV file and returns the average of each numeric column. Only code.'
ASK = 'What is the hostname stored in /etc/hostname? Use the read_file tool.'
with open(long_file, encoding='utf-8') as f:
    LONG_TEXT = f.read()


def user(text):
    return {'role': 'user', 'content': text}


def assistant(text):
    return {'role': 'assistant', 'content': text}


def post(path, body, stream):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {'content-type': 'application/json'})
    r = urllib.request.urlopen(req, timeout=900)
    if not stream:
        return json.load(r)
    events = []
    for line in r:
        line = line.decode().strip()
        if line.startswith('data: ') and line != 'data: [DONE]':
            events.append(json.loads(line[6:]))
    return events


def anthropic_blocks(blocks):
    out = []
    for b in blocks:
        if b['type'] == 'tool_use':
            out.append({'type': 'tool_use', 'name': b['name'], 'input': b['input']})
        elif b['type'] == 'thinking':
            out.append({'type': 'thinking', 'thinking': b['thinking']})
        else:
            out.append({'type': b['type'], 'text': b.get('text', '')})
    return out


def anthropic_stream(events):
    blocks, cur, stop, usage = [], None, None, {}
    for e in events:
        if e['type'] == 'message_start':
            usage.update(e['message'].get('usage', {}))
        elif e['type'] == 'content_block_start':
            cur = dict(e['content_block'])
            if cur['type'] == 'text':
                cur.setdefault('text', '')
            if cur['type'] == 'thinking':
                cur.setdefault('thinking', '')
            cur['_json'] = ''
            blocks.append(cur)
        elif e['type'] == 'content_block_delta':
            d = e['delta']
            if d['type'] == 'text_delta':
                cur['text'] += d['text']
            elif d['type'] == 'thinking_delta':
                cur['thinking'] += d['thinking']
            elif d['type'] == 'input_json_delta':
                cur['_json'] += d['partial_json']
        elif e['type'] == 'message_delta':
            stop = e['delta']['stop_reason']
            usage.update(e.get('usage', {}))
    for b in blocks:
        if b['type'] == 'tool_use':
            b['input'] = json.loads(b['_json'] or '{}')
    return {'content': anthropic_blocks(blocks), 'stop_reason': stop, 'usage': usage}


def anthropic(messages, stream=False, tools=False, **extra):
    body = {'model': 'local', 'max_tokens': 200, 'temperature': 0, 'stream': stream, 'messages': messages}
    body.update(extra)
    if tools:
        body['tools'] = [TOOL]
    r = post('/v1/messages', body, stream)
    if stream:
        return anthropic_stream(r)
    return {'content': anthropic_blocks(r['content']), 'stop_reason': r['stop_reason'], 'usage': r['usage']}


def openai_message(msg):
    calls = [{'name': c['function']['name'], 'arguments': c['function']['arguments']} for c in msg.get('tool_calls') or []]
    return {'content': msg.get('content'), 'reasoning': msg.get('reasoning_content'), 'tool_calls': calls}


def openai_stream(events):
    text, reasoning, calls, fin, usage = '', '', {}, None, None
    for c in events:
        usage = c.get('usage') or usage
        if not c.get('choices'):
            continue
        ch = c['choices'][0]
        d = ch['delta']
        text += d.get('content') or ''
        reasoning += d.get('reasoning_content') or ''
        for t in d.get('tool_calls', []):
            cur = calls.setdefault(t['index'], {'name': '', 'arguments': ''})
            cur['name'] += t.get('function', {}).get('name') or ''
            cur['arguments'] += t.get('function', {}).get('arguments') or ''
        fin = ch.get('finish_reason') or fin
    return {'content': text or None, 'reasoning': reasoning or None,
            'tool_calls': [calls[i] for i in sorted(calls)], 'finish_reason': fin, 'usage': usage}


def openai(messages, stream=False, tools=False, **extra):
    body = {'model': 'local', 'max_tokens': 200, 'temperature': 0, 'stream': stream, 'messages': messages}
    body.update(extra)
    if stream:
        body['stream_options'] = {'include_usage': True}
    if tools:
        body['tools'] = [OPENAI_TOOL]
    r = post('/v1/chat/completions', body, stream)
    if stream:
        return openai_stream(r)
    out = openai_message(r['choices'][0]['message'])
    out.update(finish_reason=r['choices'][0]['finish_reason'], usage=r.get('usage'))
    return out


def save(name, result):
    with open(f'{dest}/{name}.json', 'w') as f:
        json.dump(result, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write('\n')
    print(f'{name}: done', flush=True)
    return result


def answer_text(result):
    return ''.join(b.get('text', '') for b in result['content'] if b['type'] == 'text')


def group_base():
    for name, fn, stream, tools in [('anthropic', anthropic, False, False), ('anthropic-stream', anthropic, True, False),
                                    ('anthropic-tool', anthropic, False, True), ('anthropic-tool-stream', anthropic, True, True),
                                    ('openai', openai, False, False), ('openai-stream', openai, True, False),
                                    ('openai-tool', openai, False, True), ('openai-tool-stream', openai, True, True)]:
        save(name, fn([user(ASK if tools else PROMPT)], stream, tools))


def group_sampled():
    smp = {'temperature': 0.7, 'top_p': 0.8, 'seed': 42}
    for name, fn, stream in [('sampled-anthropic', anthropic, False), ('sampled-anthropic-stream', anthropic, True),
                             ('sampled-openai', openai, False), ('sampled-openai-stream', openai, True)]:
        save(name, fn([user(PROMPT)], stream, **smp))


def group_thinking():
    # A prompt that makes the model think at length, so the budget of 32 tokens runs out and the closing text is forced
    msgs = [user('Prove that the square root of 2 is irrational. Think through every step of the proof carefully.')]
    save('thinking-anthropic', anthropic(msgs, max_tokens=400, thinking={'type': 'enabled', 'budget_tokens': 32}))


def group_reuse():
    q1, q2 = 'Name three prime numbers above 100.', 'Which of them is the largest?'
    r1 = save('reuse-short-1', anthropic([user(q1)]))
    save('reuse-short-2', anthropic([user(q1), assistant(answer_text(r1)), user(q2)]))
    # Long conversation A, an unrelated one B, then A again: A comes back from the RAM stash
    a = user(LONG_TEXT[:6000] + '\n\nWhat is this code for? One sentence.')
    ra = save('reuse-long-1', anthropic([a], max_tokens=60))
    save('reuse-other', anthropic([user('Say hello in one short sentence.')], max_tokens=30))
    follow = user('In which language are the comments written? One word.')
    save('reuse-long-2', anthropic([a, assistant(answer_text(ra)), follow], max_tokens=30))


def save_answer(name, result):
    # What a hit and a cold run must share: the answer. The usage differs (cached tokens), so it stays out
    answer = {'content': result['content'], 'stop_reason': result['stop_reason'],
              'output_tokens': result['usage'].get('output_tokens')}
    return save(name, answer)


def hit_requests():
    # (kind, [(name, messages)]) in the order that makes the later ones hit what the earlier ones left
    cut = user(LONG_TEXT[:20000] + '\n\nRemember this code.')
    fork_head = LONG_TEXT[:14000]
    fork = [user(fork_head + '\n\n' + LONG_TEXT[16000 + 5000 * i:19000 + 5000 * i] + '\n\nSummarize this.') for i in range(3)]
    warm_head = LONG_TEXT[1000:15000] + '\n\nWhat is this code for?'
    exact = LONG_TEXT[2000:14000] + '\n\nList the functions.'
    return [
        # the second request continues the checkpoint at the start of the last message
        ('last-message-1', [cut, assistant('Noted.'), user('What kind of code is this? One short sentence.')]),
        ('last-message-2', [cut, assistant('Noted.'), user('Name one function defined in it.')]),
        # the third shares 14000 characters with the first two and nothing after; the second one wrote the fork
        ('fork-1', [fork[0]]), ('fork-2', [fork[1]]), ('fork-3', [fork[2]]),
        # the warm-up (max_tokens 0) stores the head of the prompt, the request continues it
        ('warm-up', [user(warm_head)]),
        ('warm-request', [user(warm_head + ' Answer in one sentence.')]),
        # the warm-up of exactly the prompt of the request: the end of the prompt, with its logits
        ('warm-exact-up', [user(exact)]),
        ('warm-exact', [user(exact)]),
    ]


def run_hit(prefix):
    # prefix 'hit-': a server with checkpoints; 'cold-': TONA_PROMPT_SLOTS=1, none (the warm-ups are skipped there)
    for name, msgs in hit_requests():
        if name.endswith('-up'):
            if prefix == 'hit-':
                post('/v1/messages', {'model': 'local', 'max_tokens': 0, 'temperature': 0, 'messages': msgs}, False)
            continue
        save_answer(prefix + name, anthropic(msgs, max_tokens=60))
    # The same prompt again: the end state of the previous request, with its logits
    save_answer(prefix + 'repeat', anthropic(hit_requests()[-1][1], max_tokens=60))


def group_hit():
    run_hit('hit-')


def group_hitcold():
    run_hit('cold-')


def ssd_conversation():
    return [user(LONG_TEXT[:20000] + '\n\nRemember this code.'), assistant('Noted.'),
            user('What kind of code is this? One short sentence.')]


def group_ssd1():
    save('ssd-1', anthropic(ssd_conversation(), max_tokens=60))


def group_ssd2():
    with open(f'{dest}/ssd-1.json') as f:
        first = json.load(f)
    text = ''.join(b.get('text', '') for b in first['content'] if b['type'] == 'text')
    msgs = ssd_conversation() + [assistant(text), user('Name one function defined in it.')]
    save('ssd-2', anthropic(msgs, max_tokens=60))


for g in groups:
    globals()['group_' + g]()
PY

run_client() { python3 -I "$client" "http://127.0.0.1:$port" "$results" scripts/bench-long-code.txt "$@"; }

if has_case base || has_case sampled || has_case thinking || has_case reuse; then
    start_server
    for g in base sampled thinking reuse; do
        if has_case "$g"; then run_client "$g"; fi
    done
    stop_server
fi
if has_case hit; then
    # Hits: a server with checkpoints, then a server without (one slot: no checkpoint is ever written)
    start_server
    run_client hit
    stop_server
    start_server TONA_PROMPT_SLOTS=1
    run_client hitcold
    stop_server
    for f in "$results"/hit-*.json; do
        name=${f##*/hit-}
        cmp -s "$f" "$results/cold-$name" || { echo "hit-${name%.json}: DIFFERS from the cold run (diff $f $results/cold-$name)"; status=1; }
    done
fi
if has_case ssd; then
    cache=$out/cache
    mkdir -p "$cache"
    start_server TONA_PROMPT_CACHE="$cache"
    run_client ssd1
    stop_server
    compgen -G "$cache/*.bin" >/dev/null || { echo "ssd: no checkpoint file was written, see $out/server.log"; exit 1; }
    start_server TONA_PROMPT_CACHE="$cache"
    run_client ssd2
    grep -h 'Checkpoints on the SSD' "$out/server.log" || true
    stop_server
fi

mkdir -p "$ref"
for f in "$results"/*.json; do
    name=$(basename "$f")
    if [[ $mode == record ]]; then
        if [[ -e $ref/$name && ${GOLDEN_FORCE:-0} != 1 ]]; then
            echo "${name%.json}: exists, kept"
        else
            cp "$f" "$ref/$name" && echo "${name%.json}: recorded"
        fi
    elif cmp -s "$ref/$name" "$f"; then
        echo "${name%.json}: identical"
    else
        echo "${name%.json}: DIFFERS (diff $ref/$name $f)"; status=1
    fi
done
exit $status
