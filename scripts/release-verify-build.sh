#!/usr/bin/env bash
# Maintainer, required before release-sign.sh: rebuild a tag on this machine and compare every asset of the draft
# release with the local result (SHA-256). A difference means the CI build and this build disagree: do not sign.
# On success it writes the proof file (tag, commit, SHA-256 of the draft's SHA256SUMS) that release-sign.sh checks; on a
# difference it removes a stale proof of this tag.
#
#   scripts/release-verify-build.sh vX.Y.Z [--work DIR] [--local DIR]
#
# Run it in a checkout of the tag with the toolchain from scripts/setup.sh (the same requirements as release-build.sh).
# --work   directory for the draft download and the local build (default: a new one in /var/tmp, not /tmp = RAM)
# --local  skip the download: compare with the release-build.sh output in DIR (for tests)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")
# shellcheck disable=SC1091
source "$here/release-common.sh"

[ $# -ge 1 ] || die "usage: release-verify-build.sh vX.Y.Z [--work DIR] [--local DIR]"
tag=$1
shift
check_tag "$tag"
work="" reference=""
while [ $# -gt 0 ]; do
    case $1 in
    --work) work=${2:?}; shift 2 ;;
    --local) reference=${2:?}; shift 2 ;;
    *) die "unknown option $1" ;;
    esac
done
need git sha256sum diff
proof=$(proof_file "$tag")
rm -f "$proof"
[ -n "$reference" ] || need gh

head_commit=$(git -C "$repo" rev-parse HEAD)
tag_commit=$(git -C "$repo" rev-parse -q --verify "refs/tags/$tag^{commit}") || die "tag $tag is not in this clone (git fetch --tags)"
[ "$head_commit" = "$tag_commit" ] || die "HEAD is $head_commit, the tag is $tag_commit: git checkout $tag first"

work=${work:-$(mktemp -d /var/tmp/tona-release-verify.XXXXXX)}
mkdir -p "$work"
if [ -z "$reference" ]; then
    reference=$work/draft
    echo "== Download the draft $tag from $TONA_RELEASE_REPO"
    gh release download "$tag" -R "$TONA_RELEASE_REPO" --dir "$reference"
fi

echo "== Local build"
TONA_JOBS=${TONA_JOBS:-4} "$here/release-build.sh" "$work/local" --version "${tag#v}" --work "$work/scratch" >"$work/build.log" ||
    { tail -20 "$work/build.log" >&2; die "local build failed, log: $work/build.log"; }

echo "== Compare SHA256SUMS"
# The draft also holds the macOS assets, built on a macOS runner from Homebrew packages that are not pinned: they
# cannot be rebuilt here and are proved by their build attestation (release-sign.sh checks it). Everything this build
# makes must match the draft exactly.
missing=$(sort "$work/local/SHA256SUMS" | comm -23 - <(sort "$reference/SHA256SUMS")) || true
if [ -z "$missing" ]; then
    echo "identical: all $(wc -l <"$work/local/SHA256SUMS") assets built here match the draft"
    other=$(comm -13 <(awk '{print $2}' "$work/local/SHA256SUMS" | sort) <(awk '{print $2}' "$reference/SHA256SUMS" | sort))
    [ -z "$other" ] || printf 'not rebuilt here (other platform, attestation only):\n%s\n' "$other"
else
    printf '%s\n' "$missing" >&2
    die "the local build differs from the draft (lines above exist only locally); do not sign"
fi
mkdir -p "$(dirname "$proof")"
printf 'tag=%s\ncommit=%s\nsha256sums=%s\n' "$tag" "$tag_commit" "$(sha256sum <"$reference/SHA256SUMS" | awk '{print $1}')" >"$proof"
echo "proof written: $proof"
