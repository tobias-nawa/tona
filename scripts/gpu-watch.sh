#!/usr/bin/env bash
# Watchdog for tona-server or tona-gpu-gen: logs a heartbeat and the GPU state to disk every second and steps in when
# the engine hangs.
# Usage: scripts/gpu-watch.sh PID HEARTBEAT-FILE [limit in s, default 15]
#
# Hang = GPU busy and the heartbeat (TONA_HEARTBEAT, renewed every layer) older than the limit, or the engine itself
# reports "waiting-for-gpu" (engine heartbeat text). Then, in order:
#   1. umr: state of all waves from the GPU (needs scripts/debug-setup.sh), without stopping the process
#   2. rocgdb: attach to the process, waves with program counter and source line
#   3. kill -9: the driver clears the queue, the desktop gets the GPU back
# Everything goes to work/hang/<time>/ (on disk, survives a reboot; every line with sync).
set -uo pipefail
pid=$1 hb=$2 limit=${3:-15}
cd "$(dirname "$0")/.." || exit 1
dir=work/hang/$(date +%Y%m%d-%H%M%S)
mkdir -p "$dir"
dev=$(dirname "$(readlink -f /sys/class/drm/card1/device/mem_info_vram_used)")
hw=$(echo "$dev"/hwmon/hwmon*)
log=$dir/watch.log

# Capture kernel messages right away (amdgpu page faults, ring timeouts), independent of the journald buffer
journalctl -k -f -n 0 -o short-precise > "$dir/kernel.log" 2>&1 &
jpid=$!
trap 'kill $jpid 2>/dev/null; sync' EXIT

echo "Watchdog: PID $pid, heartbeat $hb, limit ${limit} s, log $dir" | tee -a "$log"
fired=0
while kill -0 "$pid" 2>/dev/null; do
    now=$(date +%s%3N)
    beat=$(head -c 95 "$hb" 2>/dev/null | tr -s ' ' | tr -d '\n')
    if [[ $beat =~ ^[0-9]+ ]]; then age=$(( (now - ${beat%% *}) / 1000 )); else age=-1; fi
    busy=$(cat "$dev/gpu_busy_percent")
    vram=$(( $(cat "$dev/mem_info_vram_used") / 1048576 ))
    watt=$(( $(cat "$hw/power1_average") / 1000000 ))
    mhz=$(( $(cat "$hw/freq1_input") / 1000000 ))
    temp=$(( $(cat "$hw/temp2_input") / 1000 ))
    avail=$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)
    rss=$(awk '/VmRSS/ {printf "%.1f", $2/1048576}' "/proc/$pid/status" 2>/dev/null)
    # Time the driver evicted the process (KFD eviction: queues stopped, memory moved)
    ev=$(cat "/sys/class/kfd/kfd/proc/$pid"/stats_*/evicted_ms 2>/dev/null | head -1)
    echo "$(date +%T) gpu ${busy}% ${mhz} MHz ${watt} W ${temp}°C vram ${vram} MiB | ram free ${avail} G rss ${rss} G | evicted ${ev:-?} ms | age ${age} s | ${beat#* }" >> "$log"
    sync -d "$log" "$dir/kernel.log" 2>/dev/null

    if [ $fired = 0 ] && { { [ "$busy" -ge 50 ] && [ "$age" -ge "$limit" ]; } || [[ $beat == *waiting-for-gpu* ]]; }; then
        fired=1
        echo "$(date +%T) HANG: age ${age} s, gpu ${busy}%, ${beat#* }" | tee -a "$log"
        # umr instance of the RX 9060 XT (0 is the iGPU); GFXOFF off, otherwise umr reads only zeros
        inst=$(for i in 0 1 2 3; do sudo -n umr -i $i -lb 2>/dev/null | head -1 | grep -q amd7590 && echo $i && break; done)
        if [ -n "$inst" ]; then
            sudo -n umr -i "$inst" -go 0 > /dev/null 2>&1
            for r in GRBM_STATUS GRBM_STATUS2 CP_STAT CP_BUSY_STAT CP_CPC_STATUS CP_CPC_BUSY_STAT; do
                timeout 20 sudo -n umr -i "$inst" -O bits -r "*.*.reg$r"
            done > "$dir/umr-status.txt" 2>&1
            # Waves of this process's compute queue (KFD), read with the SQ halted
            comm=$(cat "/proc/$pid/comm")
            timeout 60 sudo -n umr -i "$inst" -uq "kfd,comm=$comm,type=compute" -O halt_waves,bits -wa uq \
                > "$dir/umr-waves.txt" 2>&1
            sudo -n umr -i "$inst" -go 1 > /dev/null 2>&1
            echo "  umr: $(grep -c 'wave[0-9]' "$dir/umr-waves.txt") Waves" | tee -a "$log"
        else
            echo "  umr is missing or cannot find the GPU (sudo scripts/debug-setup.sh)" | tee -a "$log"
        fi
        rocgdb=work/rocm-10.1/bin/rocgdb
        timeout 120 "$rocgdb" -q -batch -p "$pid" \
            -ex 'set pagination off' -ex 'info agents' -ex 'info dispatches' \
            -ex 'info threads' -ex 'thread apply all bt 4' > "$dir/rocgdb.txt" 2>&1
        echo "  rocgdb: $(grep -c 'AMDGPU Wave' "$dir/rocgdb.txt") Waves" | tee -a "$log"
        kill -9 "$pid" 2>/dev/null
        echo "$(date +%T) killed (kill -9)" | tee -a "$log"
        sync
    fi
    sleep 1
done
echo "$(date +%T) process $pid ended" | tee -a "$log"
