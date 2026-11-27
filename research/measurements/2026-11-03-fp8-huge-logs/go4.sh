#!/usr/bin/env bash
S=/var/tmp/claude/fp8text; R=/var/tmp/claude/fp8huge/run.sh
for arm in on off; do
  extra=()
  [ "$arm" = off ] && extra=(--env IE_PREFILL_FP8=0)
  echo "$arm start $(date +%T)"
  "$R" python3 -I "$S/text1m.py" "t1m-$arm" merged server --targets 1000000:200 --ctx 1048576 --env IE_DIAG=1 --env IE_SEQS=1 --env IE_PF_DEBUG=1 "${extra[@]}" > "$S/t1m-$arm.out" 2>&1
  echo "$arm rc=$? $(date +%T)"
  [ "$arm" = on ] && sleep 60
done
