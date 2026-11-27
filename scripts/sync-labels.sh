#!/bin/bash
# Creates or updates the labels of .github/labels.tsv (name, colour, description). Additive: a label that is not in the
# file is left alone, nothing is deleted. Used by scripts/github-setup.sh and by .github/workflows/labels.yml.
#
# Usage: scripts/sync-labels.sh OWNER/REPO [--apply]    (without --apply it prints the calls, the default)
set -euo pipefail

[[ $# -ge 1 && $# -le 2 && "$1" == */* ]] || { echo "usage: $0 OWNER/REPO [--apply]" >&2; exit 2; }
repo=$1
apply=0
if [[ $# -eq 2 ]]; then
    [[ "$2" == --apply ]] || { echo "usage: $0 OWNER/REPO [--apply]" >&2; exit 2; }
    apply=1
fi

while IFS=$'\t' read -r name color description; do
    [[ -z $name || $name == \#* ]] && continue
    if ((apply)); then
        gh label create "$name" --repo "$repo" --color "$color" --description "$description" --force >/dev/null
    else
        printf 'gh label create %q --repo %q --color %q --description %q --force\n' "$name" "$repo" "$color" "$description"
    fi
done <"$(dirname "$0")/../.github/labels.tsv"
