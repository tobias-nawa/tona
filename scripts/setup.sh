#!/usr/bin/env bash
# Sets up everything the engine needs to build and measure. No sudo, changes nothing on the system.
# Everything ends up in work/ (git-ignored). Safe to run repeatedly.
#
# Usage: scripts/setup.sh [--no-gpu]
#   --no-gpu  a machine without the reference GPU (any other GPU, a laptop, a CI container): installs the same pinned
#             ROCm toolchain, which `just lint` needs for clang-tidy, and the gate tools, but skips every step that needs
#             /dev/kfd or the gfx1200 device (self-test, WMMA smoke test). The engine itself cannot run there.
set -euo pipefail
cd "$(dirname "$0")/.."
NO_GPU=0
case "${1:-}" in
    "") ;;
    --no-gpu) NO_GPU=1 ;;
    *) printf 'usage: scripts/setup.sh [--no-gpu]\n' >&2; exit 2 ;;
esac
VENV=work/rocm-10.1
LOCK=toolchain/rocm-10.1.lock

say() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

say "Prerequisites"
command -v python3 >/dev/null || die "python3 missing"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "Python >= 3.11 required (tomllib)"
for t in git cmake ninja; do command -v $t >/dev/null || die "$t missing"; done
[ -e /usr/include/openssl/evp.h ] || pkg-config --exists openssl 2>/dev/null || die "OpenSSL development files missing (Fedora: sudo dnf install openssl-devel)"
if [ "$NO_GPU" = 0 ]; then
    [ -e /dev/kfd ] || die "/dev/kfd missing: amdgpu driver with KFD required (no AMD GPU here: scripts/setup.sh --no-gpu)"
    [ -r /dev/kfd ] && [ -w /dev/kfd ] || die "no access to /dev/kfd: add the user to the render group"
fi
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 12 ] || die "only ${free_gb} GB free, ROCm needs ~7 GB"
echo "ok"

say "ROCm 10.1 into $VENV (pinned in $LOCK)"
[ -x $VENV/bin/python ] || python3 -m venv $VENV
$VENV/bin/pip install --quiet --require-hashes -r $LOCK
$VENV/bin/pip list 2>/dev/null | grep -E '^rocm'

if [ "$NO_GPU" = 0 ]; then
    say "Self-test of the ROCm packages"
    # shellcheck disable=SC1091
    source toolchain/env.sh
    rocm-sdk test 2>&1 | tail -1
    rocminfo | grep -q "Name: *$TONA_GPU_ARCH" || die "no $TONA_GPU_ARCH found (rocminfo)"

    say "Smoke test on the GPU: WMMA"
    mkdir -p work/build
    # shellcheck disable=SC2086  # the flag variables hold several words and must be split
    $TONA_HIPCXX $TONA_HIPFLAGS -O2 bench/micro/smoke_wmma.hip -o work/build/smoke_wmma $TONA_HIPLDFLAGS -Wall -Werror
    work/build/smoke_wmma
else
    say "No GPU: self-test and WMMA smoke test skipped (--no-gpu)"
fi

say "Code gates: pre-commit into work/venv-tools (pinned in tools/gates/requirements.txt)"
TOOLS=work/venv-tools
[ -x $TOOLS/bin/python ] || python3 -m venv $TOOLS
$TOOLS/bin/pip install --quiet -r tools/gates/requirements.txt
# Hook tools (ruff, clang-format, shellcheck, lizard, gitleaks) are fetched by pre-commit on first use, pinned
# in .pre-commit-config.yaml. Skipped outside a git checkout. Hooks are shared by all worktrees of the repository.
if [ -e .git ]; then $TOOLS/bin/pre-commit install; fi
$TOOLS/bin/pre-commit --version

say "Harness tests"
(cd bench && python3 -m unittest discover -s tests -t . 2>&1 | tail -1)

say "Done. Activate the toolchain: source toolchain/env.sh"
