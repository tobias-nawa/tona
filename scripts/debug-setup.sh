#!/usr/bin/env bash
# Prepare the machine for hunting GPU hangs (once, with sudo). Undo: sudo scripts/debug-setup.sh --off
#
# - umr: reads the wave state straight from the GPU on a hang (scripts/gpu-watch.sh calls it)
# - sudoers: umr only, without a password, so the watcher can call it when it matters
# - journald writes to disk every 5 s instead of every 5 min (the last minutes were lost that way once)
# - kernel: SysRq fully on (Alt+PrintScreen+S/U/B on a black screen), panic on soft/hard lockup and reboot after
#   30 s; the kernel messages before that end up in /var/lib/systemd/pstore through efi_pstore
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "please use sudo: sudo $0 $*"; exit 1; }

user=${SUDO_USER:?run it through sudo as your own user}
sysctl_f=/etc/sysctl.d/90-tona-debug.conf
journal_f=/etc/systemd/journald.conf.d/90-tona-debug.conf
sudoers_f=/etc/sudoers.d/90-tona-umr

if [ "${1:-}" = "--off" ]; then
    rm -f "$sysctl_f" "$journal_f" "$sudoers_f"
    sysctl -q kernel.softlockup_panic=0 kernel.hardlockup_panic=0 kernel.panic=0 kernel.sysrq=16
    systemctl restart systemd-journald
    echo "reverted (umr stays installed: dnf remove umr)"
    exit 0
fi

dnf install -y -q umr

cat > "$sysctl_f" <<'CONF'
# scripts/debug-setup.sh: hunting GPU hangs
kernel.sysrq = 1
kernel.softlockup_panic = 1
kernel.hardlockup_panic = 1
kernel.panic = 30
CONF
sysctl -q -p "$sysctl_f"

mkdir -p "$(dirname "$journal_f")"
cat > "$journal_f" <<'CONF'
# scripts/debug-setup.sh: hunting GPU hangs
[Journal]
SyncIntervalSec=5s
CONF
systemctl restart systemd-journald

echo "$user ALL=(root) NOPASSWD: $(command -v umr)" > "$sudoers_f"
chmod 440 "$sudoers_f"
visudo -cq -f "$sudoers_f"

echo "done:"
echo "  umr $(umr --version 2>/dev/null | head -1)"
echo "  sysrq=$(sysctl -n kernel.sysrq) softlockup_panic=$(sysctl -n kernel.softlockup_panic) panic=$(sysctl -n kernel.panic)"
echo "  journald SyncIntervalSec=5s, sudo -n umr for $user"
