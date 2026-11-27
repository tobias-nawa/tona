#!/usr/bin/env bash
# Which areas a change touches, as `cpp=true|false`, `py=...`, `ci=...` lines (appended to $GITHUB_OUTPUT when set).
# usage: scripts/ci-changes.sh EVENT BASE HEAD
# The heavy CI jobs read these lines (the `changes` job of each workflow). Anything but a pull request, and a pull
# request that changes CI itself, turns every area on.
set -euo pipefail

[[ $# -eq 3 ]] || { echo "usage: $0 EVENT BASE HEAD" >&2; exit 2; }
event=$1
base=$2
head=$3

cpp=false
py=false
ci=false

if [[ "$event" != pull_request ]]; then
    cpp=true py=true ci=true
else
    # Three dots: the changes of the head since the merge base, not what the base gained meanwhile
    mapfile -t files < <(git diff --name-only "$base...$head")
    for f in "${files[@]}"; do
        case "$f" in
            .github/workflows/* | .github/codeql/* | .pre-commit-config.yaml | scripts/ci-changes.sh | scripts/github-setup.sh | tools/gates/*)
                ci=true ;;
        esac
        case "$f" in
            engine/* | installer/* | toolchain/* | release/* | data/* | justfile | .clang-tidy | .clang-format | CMakeLists.txt | */CMakeLists.txt | *.cmake | scripts/*.sh)
                cpp=true ;;
        esac
        # Everything that is not documentation can be read by a Python test or a gate: the tests read data/, scripts/,
        # toolchain/ and release/
        case "$f" in
            docs/* | research/* | communication/* | *.md | LICENSE | licenses/*) ;;
            *) py=true ;;
        esac
    done
fi

printf 'cpp=%s\npy=%s\nci=%s\n' "$cpp" "$py" "$ci" | tee -a "${GITHUB_OUTPUT:-/dev/null}"
