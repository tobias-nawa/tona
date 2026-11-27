#!/usr/bin/env bash
# Build the release assets from a clean copy of HEAD (git archive, never the working tree) and write them flat into
# OUTDIR: the programs, the ROCm runtime libraries, the licence texts, SHA256SUMS and an unsigned manifest.json.
# The same script runs in CI (.github/workflows/release.yml) and on the maintainer's machine (release-verify-build.sh),
# and both must produce the same bytes. Needs the toolchain from scripts/setup.sh at the commit being built.
#
#   scripts/release-build.sh OUTDIR [--version X.Y.Z] [--seq N] [--expires-days D] [--changelog-file FILE] [--work DIR]
#
# --version  default: the tag at HEAD without the leading v
# --seq      default 1; the maintainer sets the real value when signing (release-sign.sh)
# --changelog-file  file with the release notes; default: CHANGELOG.md of HEAD. The section `## [X.Y.Z]` of the version
#            goes into the manifest (the update dialog of tona, covered by the signature; the workflow
#            copies it from there into the GitHub release); a missing section stops the script before the long build starts
#            (scripts/release_notes.py). Only for the dry run does another file make sense.
# --work     scratch directory for source copy and builds (default OUTDIR.work, removed at the end unless
#            TONA_KEEP_WORK=1). Not /tmp, which is RAM here: the builds need several GB.
# Environment: TONA_JOBS (default 4) parallel compile jobs.
set -euo pipefail
export LC_ALL=C TZ=UTC

die() { printf 'release-build: %s\n' "$*" >&2; exit 1; }

repo=$(cd "$(dirname "$0")/.." && pwd)
[ $# -ge 1 ] || die "usage: release-build.sh OUTDIR [--version X.Y.Z] [--seq N] [--expires-days D] [--changelog-file FILE] [--work DIR]"
out=$1
shift
version="" seq=1 expires_days=180 changelog_file="" work=""
while [ $# -gt 0 ]; do
    case $1 in
    --version) version=${2:?}; shift 2 ;;
    --seq) seq=${2:?}; shift 2 ;;
    --expires-days) expires_days=${2:?}; shift 2 ;;
    --changelog-file) changelog_file=${2:?}; shift 2 ;;
    --work) work=${2:?}; shift 2 ;;
    *) die "unknown option $1" ;;
    esac
done

cd "$repo"
git rev-parse --verify -q HEAD >/dev/null || die "not a git checkout"
commit=$(git rev-parse HEAD)
if [ -z "$version" ]; then
    tag=$(git describe --tags --exact-match HEAD 2>/dev/null) || die "HEAD has no tag; pass --version"
    version=${tag#v}
fi
# Release notes first: a version without them must not cost a build
if [ -n "$changelog_file" ]; then
    notes=$(python3 -I "$repo/scripts/release_notes.py" "$version" --file "$changelog_file") || exit 1
else
    notes=$(git show HEAD:CHANGELOG.md | python3 -I "$repo/scripts/release_notes.py" "$version" --file -) || exit 1
fi
# Timestamps that a tool might embed come from the commit, not from the clock
SOURCE_DATE_EPOCH=$(git log -1 --format=%ct HEAD)
export SOURCE_DATE_EPOCH

mkdir -p "$out"
out=$(cd "$out" && pwd)
[ -z "$(ls -A "$out")" ] || die "$out is not empty"
work=${work:-$out.work}
[ ! -e "$work" ] || die "$work exists; remove it or pass another --work"
mkdir -p "$work"
work=$(cd "$work" && pwd)
src=$work/src engine_build=$work/engine
mkdir -p "$src"
if [ "${TONA_KEEP_WORK:-0}" != 1 ]; then trap 'rm -rf "$work"' EXIT; fi

echo "== Source: git archive $commit"
git archive --format=tar HEAD | tar -x -C "$src"

echo "== Toolchain (must match toolchain/rocm-10.1.lock of this commit)"
lock_sha=$(sha256sum "$src/toolchain/rocm-10.1.lock" | cut -d' ' -f1)
[ "$lock_sha" = "$(sha256sum toolchain/rocm-10.1.lock | cut -d' ' -f1)" ] ||
    die "the toolchain in $repo/work was installed from another rocm-10.1.lock than HEAD has; check out the commit and run scripts/setup.sh"
# shellcheck disable=SC1091
source toolchain/env.sh
lock12=${lock_sha:0:12}

jobs=${TONA_JOBS:-4}
echo "== Engine (TONA_RELEASE=ON, -j$jobs)"
cmake -S "$src/engine" -B "$engine_build" -G Ninja -DTONA_RELEASE=ON -DTONA_MODEL=/nonexistent/model.gguf \
    -DTONA_LLAMACPP_BUILD=/nonexistent >/dev/null
cmake --build "$engine_build" -j"$jobs" --target tona-server tona


echo "== Assets into $out"
declare -a specs=()
declare -a binaries=()
add_program() { # SOURCE ASSET PATH
    install -m 0755 "$1" "$out/$2"
    specs+=("--file" "$2:${3:-bin/$2}:0755:linux-x86_64-gfx1200")
    # Every compiled program of the release is checked below, whatever it is called; scripts are not ELF files
    if [ "$(head -c4 "$1" | tail -c3)" = ELF ]; then binaries+=("$out/$2"); fi
}
add_program "$engine_build/tona-server" tona-server
# The one program of the user: setup, control centre, commands, coding agents and updater. Its asset carries the platform
# in the name because it is the file a user downloads first (docs/use/install.md); inside the release it is bin/tona.
add_program "$engine_build/tona" linux-x86_64-gfx1200-tona bin/tona

# The hardening of engine/cmake/hardening.cmake is in every compiled program of the release (PIE, full RELRO, canary,
# FORTIFY, CET): the programs are the ones just added, not a list of names kept here
[ "${#binaries[@]}" -gt 0 ] || die "no compiled program among the release assets"
python3 -I "$src/tools/check_hardening.py" "${binaries[@]}" || die "a release program is not hardened (see above)"

# Runtime libraries: whatever tona-server loads from the ROCm wheels. Libraries of the Fedora base system (libc,
# libstdc++, libm, libgcc_s) are not shipped. Wheel files are fixed by the hashes in the lock file.
# The OpenMP runtime is shipped: the host code of the HIP sources calls the LLVM one (__kmpc_*), whose libgomp.so.1 and
# libomp.so (same file, SONAME libomp.so) also serve the GOMP_* calls of the C++ sources; the system libgomp has no
# __kmpc_*. One runtime, the toolchain's (2026-11-18).
# The programs have no rpath to the toolchain, so ldd gets it from LD_LIBRARY_PATH.
# Names are flat (docs/maintain/release.md): the manifest maps them to runtime/<name>.
declare -a shipped=()
ship_runtime() {
    local name=$1 path=$2 asset=rocm-$lock12-$1
    install -m 0644 "$(readlink -f "$path")" "$out/$asset"
    specs+=("--file" "$asset:runtime/$name:0644:linux-x86_64-gfx1200")
    shipped+=("$name")
}
while read -r name path; do
    ship_runtime "$name" "$path"
done < <(LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib" ldd "$engine_build/tona-server" | awk -v root="$(readlink -f "$ROCM_PATH/..")" '
    $3 ~ /^\// { cmd = "readlink -f " $3; cmd | getline real; close(cmd); if (index(real, root) == 1) print $1, $3 }' | sort)
# A needed name that ldd did not list because another name of the same file came first (libomp.so is the SONAME of
# libgomp.so.1): ship it under its own name too, so no lookup depends on the order of loading
while read -r name; do
    [[ " ${shipped[*]} " == *" $name "* ]] && continue
    [ -f "$ROCM_PATH/lib/llvm/lib/$name" ] && ship_runtime "$name" "$ROCM_PATH/lib/llvm/lib/$name"
done < <(readelf -d "$engine_build/tona-server" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')
# Every needed library is shipped or belongs to the base system; fails for a forgotten one
python3 "$src/scripts/release_needed.py" "$engine_build/tona-server" "${shipped[@]}" ||
    die "tona-server needs a library that the release does not ship (see above)"
# No program carries an absolute RUNPATH (a link flag once leaked the toolchain's directories into it)
python3 "$src/scripts/release_needed.py" --runpath-only "$engine_build/tona" ||
    die "a program has a RUNPATH entry that is not relative to \$ORIGIN (see above)"

# Licence texts of the shipped ROCm parts, one file, in this order: the overview table, the vendored texts of
# licenses/ (LLVM, the third-party code inside the libraries, the sysdeps libraries) and the licence files that the
# wheels carry for hip, comgr, rocr and rocprofiler-register. The wheels have none for LLVM proper (their LLVM file has
# 6 lines and is not the licence), none for the code statically linked into the AMD libraries (libhsakmt, ELFIO, glog,
# fmt, device libraries) and a NOTICES.txt that covers software we do not ship (rocgdb, bash, glibc), so those two
# are not used. The vendored texts are checked in, never downloaded at build time (docs/maintain/release.md).
docs=$ROCM_PATH/share/doc
license=rocm-$lock12-LICENSES.txt
{
    cat "$src/licenses/overview.txt"
    for f in licenses/llvm-LICENSE.TXT licenses/llvm-third-party.txt licenses/rocm-extra.txt licenses/rocm-sysdeps.txt; do
        [ -f "$src/$f" ] || die "licence file $f missing in the source"
        printf '\n==== %s ====\n\n' "$f"
        cat "$src/$f"
    done
    for f in "$docs"/hip/LICENSE.md "$docs"/amd_comgr/LICENSE.txt "$docs"/rocr/LICENSE.md \
        "$docs"/rocprofiler-register/LICENSE.md; do
        [ -f "$f" ] || die "licence file $f missing in the toolchain"
        printf '\n==== %s ====\n\n' "${f#"$ROCM_PATH"/}"
        cat "$f"
    done
} >"$out/$license"
chmod 0644 "$out/$license"
specs+=("--file" "$license:runtime/LICENSES.txt:0644:linux-x86_64-gfx1200")

# Licence of Tona and of the code compiled into its programs (cpp-httplib, nlohmann/json, FTXUI, the
# parts ported from llama.cpp/ggml): MIT asks for the notice in every copy, binaries included (engine/third_party/README.md).
own=tona-LICENSES.txt
{
    printf '==== LICENSE (Tona) ====\n\n'
    cat "$src/LICENSE"
    for f in "$src"/engine/third_party/LICENSE-*; do
        printf '\n==== %s ====\n\n' "${f#"$src"/}"
        cat "$f"
    done
} >"$out/$own"
chmod 0644 "$out/$own"
specs+=("--file" "$own:runtime/tona-LICENSES.txt:0644:linux-x86_64-gfx1200")

echo "== Manifest and SHA256SUMS"
python3 -I "$repo/scripts/release_manifest.py" create --dir "$out" --output "$out/manifest.json" \
    --version "$version" --seq "$seq" --expires-days "$expires_days" --source-date-epoch "$SOURCE_DATE_EPOCH" \
    --commit "$commit" --lock-sha256 "$lock_sha" --changelog "$notes" "${specs[@]}"
sums=$(cd "$out" && find . -maxdepth 1 -type f ! -name manifest.json -printf '%f\n' | sort | xargs sha256sum)
printf '%s\n' "$sums" >"$out/SHA256SUMS"
echo "done: $(find "$out" -maxdepth 1 -type f | wc -l) files, version $version, commit $commit"
