#!/bin/bash
# tona-server as a systemd user service (macOS: a background process, no service yet). Called through just: just start | stop | install-service | uninstall-service
#   start      one is already running: nothing; installed service: systemctl --user start; otherwise a transient
#              service (systemd-run) that disappears with stop
#   stop       stop the service; if the server runs in a terminal (just serve): SIGTERM to its PID from /status
#   install    set the service up permanently, starts at login
#   uninstall  remove it again
# Settings from the environment (just sets them, also from work/tona.conf): TONA_MODEL, TONA_DRAFTER, TONA_PORT, TONA_CTX,
# TONA_PROFILE, TONA_ARGS, TONA_API_KEY_FILE (default ~/.config/tona/api-key, created on demand; none: no key);
# also passed on to the server: TONA_SEQS, TONA_RAM_GB, TONA_VRAM_GB, TONA_OEA_K0, TONA_ROLE_EFFORT, TONA_CLOCK_HOLD, TONA_PROMPT_CACHE_GB,
# TONA_PREFILL_FP8, TONA_DUMP_DIR.
# TONA_DIAG=1 (a developer's machine): allocation trace (one file per PID) and GPU core dump at a crash, in TONA_DIAG_DIR
# (default work/diag); toolchain/env.sh does the same for every other GPU process, scripts/diag-clean.sh the cleaning
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
# ExecStartPre of the installed service, without the server settings
if [ "${1:-}" = rotate-diag ]; then
    "$repo/scripts/diag-clean.sh" "${TONA_DIAG_DIR:?}"
    exit 0
fi
unit=tona-server
file=$HOME/.config/systemd/user/$unit.service
port=${TONA_PORT:-8662}
# Memory limit as in just serve: TONA_RAM_GB (page cache included), else RAM minus 6 GB for the rest of the system
if [ -n "${TONA_RAM_GB:-}" ]; then
    limit=$(awk "BEGIN {printf \"%d\", $TONA_RAM_GB * 1e9}")
elif [ "$(uname -s)" = Darwin ]; then
    limit=$(( $(sysctl -n hw.memsize) / 1073741824 - 6 ))G
else
    limit=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1048576 - 6 ))G
fi
envs=(ROCR_VISIBLE_DEVICES=0 "TONA_EXPERT_PROFILE=${TONA_PROFILE:-$repo/work/profile/experts-coder.bin}"
      "TONA_PROMPT_CACHE=$repo/work/prompt-cache")
for k in TONA_SEQS TONA_RAM_GB TONA_VRAM_GB TONA_OEA_K0 TONA_ROLE_EFFORT TONA_CLOCK_HOLD TONA_PROMPT_CACHE_GB TONA_PREFILL_FP8 TONA_DUMP_DIR; do
    [ -n "${!k:-}" ] && envs+=("$k=${!k}")
done
# A rare fault can only be assigned to a buffer with the trace of the run that crashed and the GPU dump, so
# both are on for every start (research/measurements/2026-10-31-moe-sum-fault.md). The trace costs one line per
# allocation, one file per PID, so no start overwrites another's; the dump (tens of GB) is written only at a crash
# and needs a core limit above 0.
diag=""
if [ -n "${TONA_DIAG:-}" ] && [ "$TONA_DIAG" != 0 ]; then
    diag=${TONA_DIAG_DIR:-$repo/work/diag}
    envs+=("TONA_ALLOC_TRACE=$diag/alloc-trace-%p.txt" "TONA_KFD_SMI_LOG=$diag/kfd-smi-%p.log" "HSA_COREDUMP_PATTERN=$diag/gpucore-%p.core")
fi
# Key for /v1/* (tona reads it for Claude Code); if missing, one is generated. Without: TONA_API_KEY_FILE=none
keyfile=${TONA_API_KEY_FILE:-$HOME/.config/tona/api-key}
if [ "$keyfile" != none ] && [ ! -s "$keyfile" ]; then
    mkdir -p "$(dirname "$keyfile")"
    (umask 077 && head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$keyfile")
fi
keyarg=""
[ "$keyfile" != none ] && keyarg="--api-key-file $keyfile"
cmd="$repo/work/build/engine/tona-server ${TONA_MODEL:?} ${TONA_DRAFTER:?} --port $port --ctx ${TONA_CTX:-262144} $keyarg ${TONA_ARGS:-}"
status() { curl -sf -m 1 "http://127.0.0.1:$port/status"; }
# Port of the running service (empty: none active); stop and start only touch a service on their own port
unit_port() {
    command -v systemctl >/dev/null || return 0
    systemctl --user -q is-active $unit 2>/dev/null || return 0
    systemctl --user show $unit -p ExecStart --value | grep -o -- '--port [0-9]*' | head -1 | cut -d' ' -f2
}

if [ "$(uname -s)" = Darwin ] && { [ "${1:-}" = install ] || [ "${1:-}" = uninstall ]; }; then
    echo "no service on macOS yet (launchd): just start runs the server until just stop" >&2
    exit 1
fi
case "${1:-}" in
start)
    if status >/dev/null; then echo "already running on port $port"; exit 0; fi
    if p=$(unit_port) && [ -n "$p" ]; then
        echo "tona-server already runs as a service on port $p; only one fits on the GPU (first: just port=$p stop)" >&2
        exit 1
    fi
    if [ "$(uname -s)" = Darwin ]; then
        # no systemd (launchd comes with the release): a background process that just stop ends via /status
        log=$repo/work/tona-server.log
        cd "$repo"
        # shellcheck disable=SC2086  # cmd is deliberately split at spaces into the command line
        nohup env "${envs[@]}" $cmd >"$log" 2>&1 </dev/null &
        echo "started on port $port (progress: tona, log: $log)"
        exit 0
    elif [ -f "$file" ]; then
        systemctl --user start $unit
    else
        systemctl --user reset-failed $unit 2>/dev/null || true
        if [ -n "$diag" ]; then mkdir -p "$diag"; "$repo/scripts/diag-clean.sh" "$diag"; fi
        args=()
        for e in "${envs[@]}"; do args+=(-E "$e"); done
        if [ -n "$diag" ]; then args+=(-p LimitCORE=infinity); fi
        # shellcheck disable=SC2086  # cmd is deliberately split at spaces into the command line
        systemd-run --user --unit=$unit --collect -q -p MemoryMax="$limit" -p MemorySwapMax=0 \
            --working-directory="$repo" "${args[@]}" $cmd
    fi
    echo "started on port $port (progress: tona, log: journalctl --user -u $unit -f)"
    ;;
stop)
    if [ "$(unit_port)" = "$port" ]; then
        systemctl --user stop $unit
        echo "stopped"
    elif pid=$(status | sed -n 's/.*"pid":\([0-9]*\).*/\1/p') && [ -n "$pid" ]; then
        kill "$pid"
        echo "SIGTERM sent to PID $pid"
    else
        echo "no server on port $port"
    fi
    ;;
install)
    if systemctl --user -q is-active $unit && [ ! -f "$file" ]; then systemctl --user stop $unit; fi
    mkdir -p "$(dirname "$file")"
    {
        echo "# generated by scripts/tona-service.sh; to change: just install-service with other settings"
        echo "[Unit]"
        echo "Description=tona-server (local LLM engine, port $port)"
        echo "StartLimitIntervalSec=600"
        echo "StartLimitBurst=3"
        echo "[Service]"
        echo "WorkingDirectory=$repo"
        # systemd expands %p in a unit file (to the unit prefix), so a literal %p is written as %%p
        for e in "${envs[@]}"; do echo "Environment=${e//%/%%}"; done
        if [ -n "$diag" ]; then
            echo "ExecStartPre=$repo/scripts/tona-service.sh rotate-diag"
            echo "Environment=TONA_DIAG_DIR=$diag"
            echo "LimitCORE=infinity"
        fi
        echo "ExecStart=$cmd"
        echo "MemoryMax=$limit"
        echo "MemorySwapMax=0"
        echo "Restart=on-failure"
        echo "RestartSec=30"
        echo "[Install]"
        echo "WantedBy=default.target"
    } > "$file"
    systemctl --user daemon-reload
    systemctl --user enable --now $unit
    echo "installed: $file, starts at login from now on"
    ;;
uninstall)
    systemctl --user disable --now $unit 2>/dev/null || true
    rm -f "$file"
    systemctl --user daemon-reload
    echo "removed"
    ;;
*)
    echo "Usage: $0 start|stop|install|uninstall" >&2
    exit 2
    ;;
esac
