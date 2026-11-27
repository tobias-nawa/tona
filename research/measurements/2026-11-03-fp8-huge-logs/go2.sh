#!/usr/bin/env bash
S=/var/tmp/claude/fp8huge; R=$S/run.sh
h() { echo "$1 $(date +%T)"; local l=$1; shift; "$R" python3 -I "$S"/curve.py "$l" merged server --targets 512000:200 --ctx 1048576 --env IE_DIAG=1 --env IE_SEQS=1 --env IE_PF_DEBUG=1 "$@" > "$S/$l".out 2>&1; echo "rc=$? $(date +%T)"; }
h on200
sleep 60
h off200 --env IE_PREFILL_FP8=0
echo "done $(date +%T)"
