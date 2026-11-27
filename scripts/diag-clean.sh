#!/bin/bash
# diag-clean.sh DIR: keep the disk of a diagnostic directory (TONA_DIAG=1) bounded. Shared by toolchain/env.sh and
# scripts/tona-service.sh. A GPU dump is about 45 GB, so only the newest gpucore-*.core stays; traces are small (one
# line per allocation) and go after 7 days, which is longer than a fault takes to be noticed.
# Silent, and never fails: it runs from env.sh, which is sourced into every build and measurement script.
d=${1:-}
[ -d "$d" ] || exit 0
cores=()
mapfile -t cores < <(ls -t "$d"/gpucore-*.core 2>/dev/null || true)
if [ ${#cores[@]} -gt 1 ]; then rm -f -- "${cores[@]:1}" 2>/dev/null || true; fi
find "$d" -maxdepth 1 \( -name 'alloc-trace*.txt' -o -name 'kfd-smi-*.log' \) -mtime +7 -delete 2>/dev/null || true
exit 0
