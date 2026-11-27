# shellcheck shell=bash
# Source this at the top of a script that uses the GPU, before `cd` and before the arguments are consumed:
#   source "$(dirname "$0")/gpu-lock.sh"
# Without TONA_GPU_LOCK_HELD it restarts the script through scripts/gpu_queue.py (a ticket line in front of the flock) on
# TONA_GPU_LOCK (default work/gpu.lock of the checkout), the lock every GPU run of this machine shares (docs/contribute/testing.md). The variable is inherited, so a script that another one
# started (golden.sh under gpu-status.sh) does not take the lock a second time. scripts/ab.py follows the same rule.
# An outer `flock LOCK script` (older tickets and scripts) would deadlock the restart: when the lock is taken, the
# script looks for an ancestor that holds it (/proc/locks) and then goes on without a new lock.
# Not for `-h`/`--help`: they start nothing.

# _tona_lock_holder_ancestor FILE: PID of an ancestor of this shell that holds a flock on FILE, else nothing
_tona_lock_holder_ancestor() {
    local inode holders pid
    inode=$(stat -c %i "$1" 2>/dev/null) || return 0
    # /proc/locks: "1: FLOCK  ADVISORY  WRITE <pid> <major>:<minor>:<inode> 0 EOF"; the major:minor is hex
    holders=$(awk -v ino="$inode" '$2 == "FLOCK" { n = split($6, f, ":"); if (f[n] == ino) print $5 }' /proc/locks 2>/dev/null)
    [[ -n $holders ]] || return 0
    pid=$$
    while [[ $pid -gt 1 ]]; do
        if grep -qx "$pid" <<<"$holders"; then echo "$pid"; return 0; fi
        pid=$(awk '$1 == "PPid:" { print $2 }' "/proc/$pid/status" 2>/dev/null)
        [[ -n $pid ]] || return 0
    done
}

case ${1:-} in
    -h | --help) ;;
    *)
        if [[ -z ${TONA_GPU_LOCK_HELD:-} ]]; then
            _tona_lock=${TONA_GPU_LOCK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/work/gpu.lock}
            mkdir -p "$(dirname "$_tona_lock")" 2>/dev/null
            # A probe that takes and drops the lock at once; if it fails, someone holds the lock: maybe a parent
            if ! flock -n "$_tona_lock" true 2>/dev/null; then
                _tona_parent=$(_tona_lock_holder_ancestor "$_tona_lock")
                if [[ -n $_tona_parent ]]; then
                    echo "GPU lock held by parent $_tona_parent" >&2
                    export TONA_GPU_LOCK_HELD=1
                else
                    echo "waiting for the GPU lock $_tona_lock ..." >&2
                fi
            fi
            # In line behind earlier callers (first come, first served), then under the flock (scripts/gpu_queue.py)
            [[ -n ${TONA_GPU_LOCK_HELD:-} ]] || exec env TONA_GPU_LOCK="$_tona_lock" python3 \
                "$(dirname "${BASH_SOURCE[0]}")/gpu_queue.py" --label "${0##*/}" -- "$0" "$@"
        fi
        ;;
esac
