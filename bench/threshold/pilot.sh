#!/bin/bash
# pilot.sh: the pilot runs, each variant under its own hold of the GPU lock (README.md, "Run").
# Usage: pilot.sh LABEL [VAR=value ...] -- extra run.py arguments
set -u
dir=$(cd "$(dirname "$0")" && pwd)
W76_WORK=${W76_WORK:-$dir/../../work/tmp/w76}
label=$1
shift
envs=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
[ $# -gt 0 ] && shift
log=$W76_WORK/logs/$label-server.log
flock "${TONA_GPU_LOCK:-$dir/../../work/gpu.lock}" env "${envs[@]}" "$dir/with_server.sh" "$log" "${CTX:-69632}" \
  sh -c "python3 -I $dir/run.py http://127.0.0.1:\$PORT $label pydash,boltons $*" \
  >"$W76_WORK/logs/$label.out" 2>&1
python3 -I "$dir/score.py" "$label" >>"$W76_WORK/logs/$label.out" 2>&1
