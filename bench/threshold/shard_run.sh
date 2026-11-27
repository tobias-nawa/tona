#!/bin/bash
# shard_run.sh: the task benchmark on two shards of the same model (old pin and the shard of 2026-10-19), one build,
# the same tasks, no variant switches. Each run holds the GPU lock on its own (pilot.sh); finished labels are skipped.
# Environment: BUILD (default engine), W76_PORT (default 8091), W76_WORK, TONA_GPU_LOCK.
# Labels in the order they run: old-ref1 / new-ref1 (the pair, first because they answer the question), old-ref2
# (TONA_DRAFT_N=1: noise floor f0 and selection run), old-closed (A only, repository removed: selection rule), new-ref2
# (f0 of the new shard). Argument: one label to run only that one.
set -u
dir=$(cd "$(dirname "$0")" && pwd)
export W76_WORK=${W76_WORK:-$dir/../../work/tmp/w76}
export BUILD=${BUILD:-engine}
export W76_PORT=${W76_PORT:-8091}
runs=$W76_WORK/runs
mkdir -p "$runs" "$W76_WORK/logs"
shard() { # old|new: model directory and the expert profile that belongs to it
  case $1 in
  old) export W76_MODEL_DIR=work/models/coder-iq1_m W76_PROFILE=work/profile/experts-coder.bin ;;
  new) export W76_MODEL_DIR=work/models/coder-iq1_m-1019 W76_PROFILE=work/profile/experts-coder-1019.bin ;;
  esac
}
step() { # LABEL SHARD [VAR=value ...] -- [run.py arguments]
  local label=$1 which=$2
  shift 2
  if [ -f "$runs/$label.done" ]; then return; fi
  shard "$which"
  "$dir/pilot.sh" "$label" "$@" || return
  touch "$runs/$label.done"
  sleep 120 # room for other workers' short runs between two holds of the lock
}
only=${1:-all}
run() { [ "$only" = all ] || [ "$only" = "$1" ] && step "$@"; }
run old-ref1 old --
run new-ref1 new --
run old-ref2 old TONA_DRAFT_N=1 --
run old-closed old -- --closed-book --families A
run new-ref2 new TONA_DRAFT_N=1 --
