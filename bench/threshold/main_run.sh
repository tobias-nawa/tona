#!/bin/bash
# main_run.sh: the stage runs of part 2 from one build (BUILD, default w76). Each run holds the GPU lock on its own;
# afterwards two minutes of waiting give other workers room for short runs. Resumable: finished labels are skipped.
set -u
dir=$(cd "$(dirname "$0")" && pwd)
W76_WORK=${W76_WORK:-$dir/../../work/tmp/w76}
export BUILD=${BUILD:-w76}
runs=$W76_WORK/runs
step() { # LABEL [VAR=value ...] -- [run.py arguments]
  local label=$1
  if [ -f "$runs/$label.done" ]; then return; fi
  "$dir/pilot.sh" "$@" || return
  touch "$runs/$label.done"
  if [ "$label" != ref1 ] && [ "$label" != ref2 ]; then
    python3 -I "$dir/check_differs.py" ref1 "$label" || { echo "ABORT: $label identical to ref1" >&2; exit 1; }
  fi
  sleep 120
}
step ref1 TONA_IDX_SHARE=all --
step s1 TONA_IDX_SHARE=0 --
step ref2 TONA_IDX_SHARE=all TONA_DRAFT_N=1 --
step s8 TONA_IDX_SHARE=0,1,3,4,6,7,9,10 --
step s6 TONA_IDX_SHARE=0,2,4,6,8,10 --
step s5 TONA_IDX_SHARE=0,2,5,7,10 --
step s4 TONA_IDX_SHARE=0,3,6,9 --
step s2 TONA_IDX_SHARE=0,6 --
step s6b TONA_IDX_SHARE=0,2,4,6,8,10 -- --shuffle 76
step s4b TONA_IDX_SHARE=0,3,6,9 -- --shuffle 76
step closed TONA_IDX_SHARE=all -- --closed-book --families A
