#!/usr/bin/env bash
S=/var/tmp/claude/fp8huge; R=$S/run.sh
echo "off1m $(date +%T)"
"$R" python3 -I "$S"/curve.py off1m merged server --targets 1000000:200 --ctx 1048576 --env IE_DIAG=1 --env IE_SEQS=1 --env IE_PF_DEBUG=1 --env IE_PREFILL_FP8=0 > "$S"/off1m.out 2>&1
echo "rc=$? done $(date +%T)"
