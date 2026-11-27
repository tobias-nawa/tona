#!/usr/bin/env bash
# One GPU job: start a server for one model, run one phase, stop the server. Run it under scripts/gpu_queue.py.
# Usage: bench/polyglot/model_job.sh MODEL PHASE [LIMIT]
#   MODEL  ornith | bolt | coder-tona | dm-tona | flash512-tona
#   PHASE  speed   bench/run cases short-code, long-decode, needle-30k, 1 run (decode and prefill rates)
#          quality polyglot exercises, up to LIMIT new ones (resumable), results in $OUT/MODEL.jsonl
#          swe     SWE-bench Verified subset (bench/swe/subset.txt), tasks START+1 .. START+LIMIT, mini-swe-agent, then grading
#          chat    the 20 fixed chat prompts (bench/chat), answers in $OUT/chat-MODEL.jsonl
#          bfcl    BFCL multi-turn cases (bench/bfcl), see run_bfcl below
# Optional environment: ENGINE_DIR (build under work/build, default engine), RAM_GB (the RAM class: TONA_RAM_GB and the
# scope's MemoryMax in decimal bytes, as the installer's unit; default 54G without TONA_RAM_GB), CASES (speed cases).
# The port is 8091 (8099 is used by another worker); the server runs in a systemd scope with a memory limit and no swap (AGENTS.md).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODEL_ID=${1:?model}
PHASE=${2:?phase}
LIMIT=${3:-1000000}
START=${START:-0}
VENV=${VENV:-/var/tmp/claude/139/venv}
VENV_BFCL=${VENV_BFCL:-/var/tmp/claude/139/venv-bfcl}
PORT=${PORT:-8091}  # 8099 is used by the other worker (checked 2026-11-15); the ticket allows 8099 or 8091
OUT=${OUT:-/var/tmp/claude/139/results}
DATA=${DATA:-/var/tmp/claude/139/polyglot-benchmark-main}
VALID=${VALID:-/var/tmp/claude/139/valid.txt}
M=$ROOT/work/models
mkdir -p "$OUT"
# shellcheck disable=SC1091
source "$ROOT/toolchain/env.sh"

# Agent suites need room for long histories (mini-swe-agent keeps all steps); speed and polyglot run at 64K
CTX=65536
case $PHASE in swe) CTX=262144 ;; bfcl) CTX=131072 ;; esac  # swe: 128K ran out after 97 steps in the first pilot task
# Same llama.cpp flags for Ornith and Bolt: experts in RAM with a 8 GiB GPU cache (scripts/baseline.sh), int8 KV.
LLAMA=(-c "$CTX" -np 1 -ngl 999 --cpu-moe --moe-cache-mib 8192 -fa on -ctk q8_0 -ctv q8_0 -t 6 --jinja --metrics --port "$PORT")
LLAMA_BIN=$ROOT/work/build/llama.cpp-hip/bin/llama-server
# Sampling from the model cards (research/measurements/2026-11-15-model-eval-ornith-bolt.md, plan)
case $MODEL_ID in
    ornith)
        cmd=("$LLAMA_BIN" -m "$M/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf" "${LLAMA[@]}")
        SAMPLING='{"temperature":1.0,"top_p":0.95,"top_k":20}' ;;
    bolt)
        cmd=("$LLAMA_BIN" -m "$M/nemotron-bolt-74b-a9b-24gb/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf" "${LLAMA[@]}")
        SAMPLING='{"temperature":1.0,"top_p":0.95}' ;;
    coder-tona|dm-tona|flash512-tona)
        gguf=$M/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
        [ "$MODEL_ID" = dm-tona ] && gguf=$M/coder-dm/coder-DM-164e-00001-of-00002.gguf
        # 512 experts per layer: the repacked ISTA file, with the Coder's MTP head as in the speed run
        [ "$MODEL_ID" = flash512-tona ] && gguf=$M/flash-next-512/dense/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
        cmd=("$ROOT/work/build/${ENGINE_DIR:-engine}/tona-server" "$gguf"
             --mtp "$M/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf" --port "$PORT" --ctx "$CTX")
        export OMP_NUM_THREADS=6 ROCR_VISIBLE_DEVICES=0
        [ -n "${RAM_GB:-}" ] && export TONA_RAM_GB="$RAM_GB"
        SAMPLING='{"temperature":1.0,"top_p":0.95,"top_k":20}' ;;
    *) echo "unknown model $MODEL_ID" >&2; exit 1 ;;
esac

# A server of the previous job may still be releasing the port; wait up to 2 minutes, then give up
for _ in $(seq 1 60); do ss -ltn | grep -q "127.0.0.1:$PORT " || break; sleep 2; done
if ss -ltn | grep -q "127.0.0.1:$PORT "; then echo "port $PORT in use" >&2; exit 1; fi
MEM_MAX=54G
[ -n "${RAM_GB:-}" ] && MEM_MAX=$(awk "BEGIN {printf \"%d\", $RAM_GB * 1e9}")
TC=$OUT/tc-$MODEL_ID-$PHASE-$(date +%H%M%S)
python3 "$ROOT/tools/hw_report.py" > "$TC.hw.md" || true
python3 "$ROOT/tools/quiet_check.py" --json > "$TC.quiet-before.json" || true
LOG=$OUT/server-$MODEL_ID-$PHASE-$(date +%H%M%S).log
systemd-run --user --scope -q -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 "${cmd[@]}" > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true' EXIT
for i in $(seq 1 900); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null && { echo "ready after $i s"; break; }
    kill -0 $PID 2>/dev/null || { echo "server exited, see $LOG" >&2; tail -20 "$LOG" >&2; exit 1; }
    sleep 1
done
curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "server not ready after 900 s" >&2; exit 1; }

case $PHASE in
    speed)
        python3 "$ROOT/bench/run" run --url "http://127.0.0.1:$PORT" --model default -c "${CASES:-short-code,long-decode,needle-30k}" \
            -n 1 -l "m139-$MODEL_ID${RAM_GB:+-ram$RAM_GB}" ;;
    quality)
        python3 -I "$ROOT/bench/polyglot/run.py" run --data "$DATA" --valid "$VALID" --url "http://127.0.0.1:$PORT" \
            --sampling "$SAMPLING" --limit "$LIMIT" --out "$OUT/$MODEL_ID.jsonl" ;;
    bfcl)
        # BFCL has no generic OpenAI endpoint; its chat-completions FC handler (registered for the MiniCPM-SALA-FC entry)
        # reads OPENAI_BASE_URL, so the entry is used as a name only. Temperature 0.001 is BFCL's own default.
        export BFCL_PROJECT_ROOT="$OUT/bfcl-$MODEL_ID" OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1" OPENAI_API_KEY=none
        mkdir -p "$BFCL_PROJECT_ROOT"
        cp "$ROOT/bench/bfcl/ids.json" "$BFCL_PROJECT_ROOT/test_case_ids_to_generate.json"
        CATS=multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context
        "$VENV_BFCL/bin/bfcl" generate --model openbmb/MiniCPM-SALA-FC --test-category "$CATS" --run-ids \
            --temperature 0.001 --num-threads 1 --result-dir result
        "$VENV_BFCL/bin/bfcl" evaluate --model openbmb/MiniCPM-SALA-FC --test-category "$CATS" --partial-eval ;;
    chat)
        python3 -I "$ROOT/bench/chat/run.py" "http://127.0.0.1:$PORT" "$OUT/chat-$MODEL_ID.jsonl" "$SAMPLING" ;;
    swe)
        ids=$(sed -n "$((START + 1)),$((START + LIMIT))p" "$ROOT/bench/swe/subset.txt" | paste -sd'|')
        mkdir -p "$OUT/msa-config"
        # executable=podman: the `docker` emulation prints a notice first, so the agent's submit marker is no longer the
        # first output line and a submitted task is not recognised (seen in the Ornith chunk 1; patches salvaged)
        # SWE_WALL_S: wall-clock cap of the whole chunk (default 5 h); tasks not finished by then are reported as not run
        # Global config dir emptied: no foreign keys or endpoints can reach the run; cost tracking off (local model)
        MSWEA_GLOBAL_CONFIG_DIR="$OUT/msa-config" MSWEA_COST_TRACKING=ignore_errors timeout "${SWE_WALL_S:-18000}" "$VENV/bin/mini-extra" swebench \
            --subset verified --split test --filter "^($ids)\$" -o "$OUT/swe-$MODEL_ID" -m openai/default \
            -c swebench.yaml -c "model.model_kwargs.api_base=http://127.0.0.1:$PORT/v1" -c model.model_kwargs.api_key=none \
            -c model.model_kwargs.temperature=1.0 -c model.model_kwargs.top_p=0.95 -c model.model_kwargs.extra_body.top_k=20 \
            -c environment.executable=podman -c environment.container_timeout=5h -c agent.step_limit="${STEP_LIMIT:-100}" -c environment.timeout=120 || true ;;  # a timeout (124) must not skip the grading below
    *) echo "unknown phase $PHASE" >&2; exit 1 ;;
esac
if [ "$PHASE" = swe ]; then
    kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true   # grading needs no server, free the GPU and RAM first
    "$VENV/bin/python" -I "$ROOT/bench/swe/evaluate.py" "$OUT/swe-$MODEL_ID/preds.json" "$OUT/swe-$MODEL_ID.eval.jsonl"
fi
kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true   # the after-check must not see our own server
sleep 3
python3 "$ROOT/tools/quiet_check.py" --compare "$TC.quiet-before.json" > "$TC.quiet-after.txt" || true
