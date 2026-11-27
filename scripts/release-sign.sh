#!/usr/bin/env bash
# Maintainer only, on the maintainer's machine: check a draft release, sign its manifest with minisign and publish.
# The CI never has the signing key; this script is the only place where a release becomes installable.
#
#   scripts/release-sign.sh vX.Y.Z --seq N [--expires-days 180] [--key FILE] [--pubkey FILE] [--severity normal|security]
#                           [--yes] [--dir DIR] [--keys-h FILE] [--local DIR [--prev-seq N] [--no-builtin-check]]
#
# Steps (all must pass, in this order; nothing is uploaded before the signature has been verified):
#   1. download the draft (gh release download) into a fresh directory
#   2. SHA256SUMS and the file list of the manifest match the files, and the manifest names the commit of this tag
#      (not --local) a proof from release-verify-build.sh: this machine rebuilt this commit and got the same SHA256SUMS
#   3. gh attestation verify for every file: built by .github/workflows/release.yml from this tag and commit
#   4. seq is greater than the seq of the newest published release
#   5. final manifest.json (seq, release time, expiry), signed with minisign (it asks for the key password)
#   6. the signature is checked against the public key that is built into tona (engine/update/keys.h)
#   7. manifest and signature are uploaded to the draft, then the release is published (asks first)
#   8. the published files are downloaded again and the signature is checked once more
#
# --key     secret key file (default: minisign's own default, ~/.minisign/minisign.key)
# --pubkey  public key file for step 6 (default: release/main.pub); use release/backup.pub with the backup key
# --keys-h  header with the compiled-in keys for step 6 (default: engine/update/keys.h); for a key replacement signed with
#           the old backup key, pass the keys.h of the previous release (git show vPREV:engine/update/keys.h)
# --local   dry run without GitHub: DIR is the output of release-build.sh. Steps 1, 3, 4 (unless --prev-seq), 7 and 8
#           are skipped; manifest.json in DIR is replaced (the draft stays in manifest.draft.json). For tests with a
#           key without password; --no-builtin-check skips the keys.h comparison of step 6 (test keys).
# --dir     working directory for the download (default: a new directory in /var/tmp, not /tmp, which is RAM here)
set -euo pipefail
export LC_ALL=C
here=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$here")
# shellcheck disable=SC1091
source "$here/release-common.sh"

tag="" seq="" expires_days=180 key="" pubkey="$repo/release/main.pub" severity="" yes=0 dir="" local_dir="" prev_seq=""
builtin_check=1
keys_h="$repo/engine/update/keys.h"
[ $# -ge 1 ] || die "usage: release-sign.sh vX.Y.Z --seq N [options], see the header of the script"
tag=$1
shift
check_tag "$tag"
while [ $# -gt 0 ]; do
    case $1 in
    --seq) seq=${2:?}; shift 2 ;;
    --expires-days) expires_days=${2:?}; shift 2 ;;
    --key) key=${2:?}; shift 2 ;;
    --pubkey) pubkey=${2:?}; shift 2 ;;
    --severity) severity=${2:?}; shift 2 ;;
    --yes) yes=1; shift ;;
    --dir) dir=${2:?}; shift 2 ;;
    --keys-h) keys_h=${2:?}; shift 2 ;;
    --local) local_dir=${2:?}; shift 2 ;;
    --prev-seq) prev_seq=${2:?}; shift 2 ;;
    --no-builtin-check) builtin_check=0; shift ;;
    *) die "unknown option $1" ;;
    esac
done
[[ $seq =~ ^[1-9][0-9]*$ ]] || die "--seq N (positive integer) is required"
[[ $expires_days =~ ^[1-9][0-9]*$ ]] || die "--expires-days must be a positive integer"
[ -f "$pubkey" ] || die "public key $pubkey not found (--pubkey)"
need minisign python3 sha256sum
[ -n "$local_dir" ] || need gh
version=${tag#v}
trusted="tona schema=1 seq=$seq version=$version"
manifest_tool=(python3 -I "$here/release_manifest.py")
sign_args=()
[ -z "$key" ] || sign_args=(-s "$key")
if [ -n "$key" ] && [ -n "$(find "$key" -perm /077 2>/dev/null)" ]; then
    echo "warning: $key is readable by other users; chmod 600 it" >&2
fi

step() { printf '\n== %s\n' "$*"; }

if [ -n "$local_dir" ]; then
    dir=$(cd "$local_dir" && pwd)
else
    dir=${dir:-$(mktemp -d /var/tmp/tona-release-sign.XXXXXX)}
    [ -z "$(ls -A "$dir")" ] || die "$dir is not empty"
    step "1. Download the draft $tag from $TONA_RELEASE_REPO into $dir"
    gh release download "$tag" -R "$TONA_RELEASE_REPO" --dir "$dir"
fi
cd "$dir"
[ -f manifest.json ] && [ -f SHA256SUMS ] || die "manifest.json or SHA256SUMS missing in $dir"
[ ! -e manifest.json.minisig ] || die "the draft already carries a signature; refusing to continue"
[ -e manifest.draft.json ] || cp manifest.json manifest.draft.json
draft=manifest.draft.json

step "2. Files, SHA256SUMS and manifest agree"
[ "$("${manifest_tool[@]}" get --manifest "$draft" version)" = "$version" ] ||
    die "the manifest in the draft is for another version than $tag"
commit=$("${manifest_tool[@]}" get --manifest "$draft" commit)
if tagged=$(git -C "$repo" rev-parse -q --verify "refs/tags/$tag^{commit}" 2>/dev/null); then
    [ "$tagged" = "$commit" ] || die "tag $tag in this clone is $tagged, the draft was built from $commit"
else
    echo "note: tag $tag is not in this clone, the commit $commit is not compared with it" >&2
fi
python3 -I - "$draft" <<'PY'
import json, os, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {entry["asset"] for entry in manifest["files"]} | {"manifest.json", "manifest.draft.json", "SHA256SUMS"}
present = set(os.listdir("."))
extra, missing = present - expected, expected - present
if extra or missing:
    sys.exit(f"unexpected files {sorted(extra)}, missing files {sorted(missing)}")
PY
sums_names=$(awk '{print $2}' SHA256SUMS | sort)
# No -printf and no stat -c: BSD find and stat on macOS have neither, and the dry run runs on a Mac too
asset_names=$(find . -maxdepth 1 -type f ! -name manifest.json ! -name manifest.draft.json ! -name SHA256SUMS | sed 's|^\./||' | sort)
[ "$sums_names" = "$asset_names" ] || die "SHA256SUMS does not list exactly the assets"
sha256sum --check --strict --quiet SHA256SUMS

if [ -z "$local_dir" ]; then
    step "2b. Local rebuild proof (Linux assets; macOS assets are not rebuildable and keep the attestation check only)"
    proof=$(proof_file "$tag")
    rebuild_cmd="scripts/release-verify-build.sh $tag"
    [ -f "$proof" ] || die "no rebuild proof for $tag: check out $tag and run $rebuild_cmd"
    [ "$(sed -n 's/^tag=//p' "$proof")" = "$tag" ] && [ "$(sed -n 's/^commit=//p' "$proof")" = "$commit" ] ||
        die "the rebuild proof is for another tag or commit than the draft ($commit): run $rebuild_cmd"
    [ "$(sed -n 's/^sha256sums=//p' "$proof")" = "$(sha256sum <SHA256SUMS | awk '{print $1}')" ] ||
        die "the draft changed since the rebuild: run $rebuild_cmd again"
    echo "ok: rebuilt locally at $commit"
    step "3. Attestations (signer $TONA_RELEASE_WORKFLOW, ref refs/tags/$tag, commit $commit)"
    for f in $asset_names manifest.json SHA256SUMS; do
        gh attestation verify "$f" --repo "$TONA_RELEASE_REPO" --signer-workflow "$TONA_RELEASE_WORKFLOW" \
            --source-ref "refs/tags/$tag" --source-digest "$commit" --deny-self-hosted-runners >/dev/null ||
            die "attestation check failed for $f"
        echo "ok $f"
    done
    step "4. seq is greater than in the newest published release"
    newest=$(gh release list -R "$TONA_RELEASE_REPO" --exclude-drafts --limit 1 --json tagName -q '.[0].tagName')
    if [ -n "$newest" ]; then
        prev_seq=$(gh release download "$newest" -R "$TONA_RELEASE_REPO" -p manifest.json -O - |
            python3 -I -c 'import json, sys; print(json.load(sys.stdin)["seq"])')
    fi
else
    step "4. seq check (local: --prev-seq only)"
fi
if [ -n "$prev_seq" ]; then
    [ "$seq" -gt "$prev_seq" ] || die "seq $seq is not greater than the published seq $prev_seq"
    echo "seq $seq > $prev_seq"
else
    echo "no earlier release: first seq is $seq"
fi

step "5. Manifest and signature"
reseal_args=(--dir . --manifest "$draft" --output manifest.json --seq "$seq" --expires-days "$expires_days")
[ -z "$severity" ] || reseal_args+=(--severity "$severity")
"${manifest_tool[@]}" reseal "${reseal_args[@]}"
minisign -S "${sign_args[@]}" -m manifest.json -t "$trusted"

step "6. Check against the built-in public key"
pub_line=$(sed -n 2p "$pubkey")
if [ "$builtin_check" = 1 ]; then
    # Only as a constant initializer (= "RW...";): the same text in a comment would not make tona trust the key
    grep -qF -- "= \"$pub_line\";" "$keys_h" ||
        die "the public key in $pubkey is not a string constant in $keys_h: tona would reject this signature"
    echo "public key is compiled into tona"
fi
minisign -V -H -p "$pubkey" -m manifest.json
[ "$(sed -n 2p manifest.json.minisig | base64 -d | head -c2)" = "ED" ] || die "the signature is not in the prehashed format (ED)"
[ "$(sed -n 3p manifest.json.minisig)" = "trusted comment: $trusted" ] || die "unexpected trusted comment"
[ "$(wc -c < manifest.json.minisig)" -le 4096 ] || die "signature above 4 KiB"

if [ -n "$local_dir" ]; then
    printf '\nDry run done: signed manifest.json and manifest.json.minisig are in %s (nothing uploaded)\n' "$dir"
    exit 0
fi

step "7. Upload and publish $tag"
gh release upload "$tag" -R "$TONA_RELEASE_REPO" --clobber manifest.json manifest.json.minisig
if [ "$yes" != 1 ]; then
    read -r -p "Publish $tag (seq $seq, expires in $expires_days days)? A published release cannot be changed. [y/N] " answer
    [ "$answer" = y ] || die "not published; the draft keeps the signed manifest"
fi
gh release edit "$tag" -R "$TONA_RELEASE_REPO" --draft=false --latest

step "8. Check the published files"
check=$(mktemp -d /var/tmp/tona-release-check.XXXXXX)
gh release download "$tag" -R "$TONA_RELEASE_REPO" --dir "$check" -p 'manifest.json*'
minisign -V -H -p "$pubkey" -x "$check/manifest.json.minisig" -m "$check/manifest.json"
cmp manifest.json "$check/manifest.json"
rm -rf "$check"
echo "released $tag"
