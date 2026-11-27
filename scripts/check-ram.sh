#!/usr/bin/env bash
# RAM check after installing new modules: modules, clock, hardware errors in the kernel log, memtester.
# Usage: scripts/check-ram.sh [MB for memtester, default 16384] [passes, default 1]
# memtester runs through sudo so it can lock the memory (mlock).
set -euo pipefail
MB=${1:-16384}
LOOPS=${2:-1}

echo "== Modules (udev, from SMBIOS)"
udevadm info -e | grep -E 'MEMORY_DEVICE_[0-9]+_(SIZE|PART_NUMBER|SPEED_MTS|CONFIGURED_SPEED_MTS)=' | sed 's/^E: //'

echo; echo "== Total"
grep -E '^(MemTotal|MemAvailable|SwapTotal|SwapFree):' /proc/meminfo

echo; echo "== Hardware errors since boot (kernel log)"
errs=$(journalctl -k -b --no-pager | grep -iE 'hardware error|machine check|mce: \[|EDAC.*(CE|UE) ' || true)
if [ -n "$errs" ]; then echo "$errs"; else echo "none"; fi

avail_mb=$(( $(awk '/^MemAvailable:/ {print $2}' /proc/meminfo) / 1024 ))
if [ "$MB" -gt $(( avail_mb - 4096 )) ]; then
    echo "ERROR: ${MB} MB requested, only ${avail_mb} MB available (4 GB reserve)" >&2
    exit 1
fi
echo; echo "== memtester ${MB} MB, ${LOOPS} pass(es) (sudo)"
start=$(date +%s)
sudo memtester "${MB}M" "$LOOPS"
echo "Duration: $(( $(date +%s) - start )) s"
