#!/usr/bin/env bash
# Join the outputs of the per-platform release builds (release-build.sh, release-build-macos.sh) into one asset
# directory with one manifest.json and one SHA256SUMS. The release workflow runs this in a job without write rights;
# release-sign.sh later reseals this manifest and signs it (docs/maintain/release.md).
#
#   scripts/release-merge.sh OUTDIR BUILD_DIR BUILD_DIR...
#
# Each BUILD_DIR holds the assets of one platform, its SHA256SUMS and its manifest.json (the files of that platform).
# The manifests must agree on version, seq, commit, notes and the other header fields (release_manifest.py merge).
set -euo pipefail
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd)

die() { printf 'release-merge: %s\n' "$*" >&2; exit 1; }

[ $# -ge 3 ] || die "usage: release-merge.sh OUTDIR BUILD_DIR BUILD_DIR..."
out=$1
shift
mkdir -p "$out"
[ -z "$(ls -A "$out")" ] || die "$out is not empty"
manifests=()
for dir in "$@"; do
    [ -f "$dir/manifest.json" ] || die "$dir/manifest.json missing"
    manifests+=(--manifest "$dir/manifest.json")
    while IFS= read -r -d '' file; do
        name=$(basename "$file")
        case $name in manifest.json | SHA256SUMS) continue ;; esac
        [ ! -e "$out/$name" ] || die "asset $name exists in two builds"
        cp -p "$file" "$out/$name"
    done < <(find "$dir" -maxdepth 1 -type f -print0)
done
python3 -I "$here/release_manifest.py" merge --output "$out/manifest.json" "${manifests[@]}"
# computed before the file exists, so that it does not list itself
# No find -printf: BSD find on macOS has none
sums=$(cd "$out" && find . -maxdepth 1 -type f ! -name manifest.json | sed 's|^\./||' | sort | xargs sha256sum)
printf '%s\n' "$sums" >"$out/SHA256SUMS"
echo "merged: $(find "$out" -maxdepth 1 -type f | wc -l) files in $out"
