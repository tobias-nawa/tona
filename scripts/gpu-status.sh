#!/usr/bin/env bash
# Runs the local golden checks that need the GPU and the model, and reports the result to GitHub as the commit status
# `gpu-golden` (a required check of the rulesets; only users with write access can set a status, so a fork cannot fake
# it). GitHub runners have no RDNA4 GPU, so this is the maintainer's step before merging a pull request.
#
# Usage: scripts/gpu-status.sh [--post] [--head REF] [--own] [--base REF] [--build DIR] [--keep-going]
#   without --post  runs the checks and only prints the status it would set
#   --post          sets the status on the checked commit with `gh api repos/{owner}/{repo}/statuses/<sha>`
#   --head REF      the commit to check (default HEAD): a pull request head fetched into this repository
#   --own           allow running from a checkout that is not part of the base branch (see below)
#   --base REF      what the pull request is compared with (default: origin/main, else main)
#   --build DIR     engine build of the checked commit (default: work/build/engine)
#   --keep-going    run every selected check even after one failed (default: cheapest first, stop at the first red,
#                   docs/contribute/measurement.md principle 4)
#
# Safe way for a pull request from someone else (docs/contribute/verification.md#gpu-check-for-a-pull-request-from-someone-else):
# run this script from a checkout of the base branch, so that it, the golden scripts, their prompts and
# tools/select_checks.py are the trusted ones, and name the pull request with --head. Only the engine build of the pull
# request is its own; read the diff before you build it. A checkout that is ahead of the base branch (a pull request
# checked out, a branch not yet merged) is refused unless you pass --own, which says: the code here is mine, or I read
# all of it.
#
# The checks are the ones docs/contribute/verification.md asks of a host-code change: golden.sh on the GPU plan and on the CPU
# plan, both prefill variants (GOLDEN_MODE=prefill, with and without the dflash drafter), and golden-server.sh (both
# APIs, MTP drafter). Kernel identity (scripts/isa-diff.py) needs a second build and stays a manual step there.
#
# tools/select_checks.py decides which of them run: the build graph of --build says which binaries a changed file goes
# into, and the binary says which golden checks it can change. There is no path list in this script. If the tool
# fails, or the build has no graph, all five run. The status description names the change class (docs/contribute/measurement.md),
# what ran, what was skipped, and the runs on the real model that this script does not do (long-context run, pressure
# series).
set -euo pipefail
# shellcheck source=/dev/null
source "$(dirname "$0")/gpu-lock.sh"
cd "$(dirname "$0")/.."

# Golden scripts and the GitHub call can be replaced to try the flow without GPU and network
GOLDEN=${GPU_STATUS_GOLDEN:-scripts/golden.sh}
GOLDEN_SERVER=${GPU_STATUS_GOLDEN_SERVER:-scripts/golden-server.sh}
GH=${GPU_STATUS_GH:-gh}

post=0
own=0
head=HEAD
base=
build=work/build/engine
keep_going=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --post) post=1 ;;
        --own) own=1 ;;
        --head) head=${2:?--head needs a ref}; shift ;;
        --base) base=${2:?--base needs a ref}; shift ;;
        --build) build=${2:?--build needs a directory}; shift ;;
        --keep-going) keep_going=1 ;;
        -h | --help) sed -n '2,/^set -euo/p' "$0" | sed '$d;s/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

if [[ -z $base ]]; then
    if git rev-parse --verify -q origin/main >/dev/null; then base=origin/main; else base=main; fi
fi
# The scripts that run here come from this checkout. If it is not part of the base branch they may be the pull
# request's, which could run its code on this machine or hang the GPU (AGENTS.md, GPU safety).
if [[ $own -eq 0 ]] && ! git merge-base --is-ancestor HEAD "$base"; then
    echo "gpu-status: this checkout ($(git rev-parse --short HEAD)) is ahead of $base, so the scripts here are not" \
        "trusted. Run from a checkout of the base branch with --head <pull request commit>, or pass --own if all of" \
        "this code is yours or read by you." >&2
    exit 2
fi
sha=$(git rev-parse --verify -q "$head^{commit}") || { echo "gpu-status: unknown commit '$head'" >&2; exit 2; }
if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
    echo "gpu-status: the working tree has uncommitted changes; run from a clean checkout" >&2
    exit 2
fi

state=success
all_goldens='{"class": "C5", "goldens": ["gpu", "cpu plan", "prefill", "prefill dflash", "server"], "skipped_count": 0, "needs": []}'
plan=$(python3 tools/select_checks.py --base "$base" --head "$sha" --build "$build" --json) || plan=
if [[ -z $plan ]]; then
    echo "gpu-status: tools/select_checks.py failed, all golden checks run" >&2
    plan=$all_goldens
else
    python3 tools/select_checks.py --base "$base" --head "$sha" --build "$build" | sed 's/^/  /'
fi
# field NAME [JOIN]: the field of the plan as text, a list joined by JOIN (default ", ")
field() { python3 -I -c 'import json,sys; v=json.load(sys.stdin)[sys.argv[1]]; print(sys.argv[2].join(v) if isinstance(v, list) else v)' "$1" "${2:-, }" <<<"$plan"; }
class=$(field class)
needs=$(field needs "; ")
skipped_count=$(field skipped_count)
mapfile -t chosen < <(field goldens $'\n')
# Cheapest first, so a red check ends the block early (run times: GOLDEN_SECONDS in tools/select_checks.py)
selected=()
for name in "prefill dflash" "gpu" "cpu plan" "server" "prefill"; do
    for c in "${chosen[@]}"; do [[ $c == "$name" ]] && selected+=("$name"); done
done
results=()
ran=()
run() {
    local name=$1
    shift
    ran+=("${name#golden }")
    if "$@"; then results+=("$name ok"); else results+=("$name FAILED"); state=failure; fi
}
not_run=()
for name in "${selected[@]}"; do
    if [[ $state == failure && $keep_going -eq 0 ]]; then not_run+=("$name"); continue; fi
    case $name in
        gpu) run "golden gpu" "$GOLDEN" compare "$build" ;;
        "cpu plan") run "golden cpu plan" env GOLDEN_PLAN=cpu "$GOLDEN" compare "$build" ;;
        # Both prefill variants run when one does: without the drafter the 13K prompt checks long-context decode, with
        # it the prefill kernels. The first was not run for days and missed a number change
        # (research/measurements/2026-11-01-golden-prefill-reference.md).
        prefill) run "golden prefill" env GOLDEN_MODE=prefill "$GOLDEN" compare "$build" ;;
        "prefill dflash") run "golden prefill dflash" env GOLDEN_MODE=prefill GOLDEN_DRAFTER=dflash "$GOLDEN" compare "$build" ;;
        server) run "golden server" "$GOLDEN_SERVER" compare "$build" ;;
        "") ;;
        *) echo "gpu-status: unknown golden check '$name' from select_checks.py" >&2; exit 2 ;;
    esac
done
if [[ ${#results[@]} -eq 0 ]]; then
    description="$class: no golden check needed"
elif [[ $state == success ]]; then
    description="$class: golden identical ($(IFS=';'; echo "${ran[*]}"))"
else
    description="$class: $(IFS=,; echo "${results[*]}")"
fi
# The runs on the real model come before the skipped checks: GitHub cuts the description at 140 characters
[[ ${#not_run[@]} -eq 0 ]] || description+="; not run after the first failure: $(IFS=,; echo "${not_run[*]}")"
[[ -z $needs ]] || description+="; needs: $needs"
[[ $skipped_count -eq 0 ]] || description+="; skipped $skipped_count"
description=${description:0:140}  # GitHub cuts a status description at 140 characters

echo "Status gpu-golden on $sha: $state - $description"
if [[ $post -eq 0 ]]; then
    echo "Dry run. With --post this would run:"
    echo "  $GH api repos/{owner}/{repo}/statuses/$sha -f state=$state -f context=gpu-golden -f description=\"$description\""
    [[ $state == success ]]
    exit
fi
"$GH" api "repos/{owner}/{repo}/statuses/$sha" -f state="$state" -f context=gpu-golden -f description="$description"
[[ $state == success ]]
