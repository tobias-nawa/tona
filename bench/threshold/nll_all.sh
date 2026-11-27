#!/bin/bash
# nll_all.sh: delta-NLL, KL and top-1 per stage (method: research/measurements/2026-10-29-indexcache.md) on py / cpp /
# sh at 32K and on the tail of the two benchmark contexts. One stage per hold of the GPU lock, two minutes of room for
# other workers in between.
set -u
dir=$(cd "$(dirname "$0")" && pwd)
W76_WORK=${W76_WORK:-$dir/../../work/tmp/w76}
export BUILD=${BUILD:-w76}
w=$W76_WORK
log=$w/logs/nll.out
texts() { # prints "TEXT PREFILL TAIL" per line
  for t in py cpp sh; do echo "$w/corp/$t.txt 32768 4096"; done
  for r in pydash boltons; do
    n=$(python3 -I -c "import json,sys; print(json.load(open('$w/ctx/$r-ctx.json'))['tokens'])")
    echo "$w/ctx/$r.txt $((n - 4096)) 4096"
  done
}
run_stage() { # TAG [VAR=value ...]
  local tag=$1
  shift
  [ -f "$w/nll/$tag.done" ] && return
  # shellcheck disable=SC2016
  flock "${TONA_GPU_LOCK:-$dir/../../work/gpu.lock}" bash -c '
    dir=$1; tag=$2; shift 2
    while read -r text pre tail; do
      for try in 1 2 3; do
        o=$("$dir/nll.sh" "$text" "$pre" "$tail" "$tag" "$@" </dev/null); echo "$o"
        echo "$o" | grep -q "tail:" && break
        sleep 20
      done
    done' _ "$dir" "$tag" "$@" < <(texts) >>"$log" 2>&1
  touch "$w/nll/$tag.done"
  sleep 120
}
run_stage ref TONA_IDX_SHARE=all
run_stage p0,1,3,4,6,7,9,10 TONA_IDX_SHARE=0,1,3,4,6,7,9,10
run_stage p0,2,4,6,8,10 TONA_IDX_SHARE=0,2,4,6,8,10
run_stage p0,2,5,7,10 TONA_IDX_SHARE=0,2,5,7,10
run_stage p0,3,6,9 TONA_IDX_SHARE=0,3,6,9
run_stage p0,6 TONA_IDX_SHARE=0,6
run_stage p0 TONA_IDX_SHARE=0
