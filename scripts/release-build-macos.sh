#!/usr/bin/env bash
# Build the macOS (Apple silicon) release assets from a clean copy of HEAD (git archive) and write them flat into
# OUTDIR with a manifest part (manifest.json: the files of macos-arm64 only). The Linux build (release-build.sh) writes
# its own part; scripts/release-merge.sh joins both into the manifest that gets signed (docs/maintain/release.md).
# The same script runs in CI (.github/workflows/release.yml, GitHub-hosted macOS runner) and on a Mac.
#
#   scripts/release-build-macos.sh OUTDIR [--version X.Y.Z] [--seq N] [--expires-days D] [--changelog-file FILE] [--work DIR]
#                                  [--allow-newer-deps]
#
# Options as in release-build.sh. --allow-newer-deps: a dry run on a Mac newer than macOS 15, whose Homebrew libomp
# and OpenSSL are built for that newer macOS; the programs then do not start on macOS 15 and must never be released. Needs Xcode or the Command Line Tools, the Metal toolchain
# (xcodebuild -downloadComponent MetalToolchain), cmake, ninja, python3 and Homebrew's libomp and openssl@3.
#
# What it ships (flat asset names, docs/maintain/release.md):
#   tona-server, tona.metallib   the server with the Metal backend linked in, and its kernel library next to it
#   tona                         setup, control centre, commands, agents and updater in one program; OpenSSL is
#                                linked statically (the Mac has no OpenSSL to rely on)
#   libomp.dylib                 Homebrew's OpenMP runtime, its install name changed to @loader_path: Apple clang ships
#                                none and a user without Homebrew would not start the server
#   LICENSES.txt                 Tona, the vendored code, libomp (LLVM) and OpenSSL
# The updater inside `tona` runs on macOS as on Linux (docs/use/update.md). Not reproducible bit for bit: the Homebrew packages are not
# pinned (docs/maintain/release.md), so the attestation of the CI build is the proof for these assets, not a local rebuild.
set -euo pipefail
export LC_ALL=C TZ=UTC

die() { printf 'release-build-macos: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "this script builds on a Mac with Apple silicon"
repo=$(cd "$(dirname "$0")/.." && pwd)
[ $# -ge 1 ] || die "usage: release-build-macos.sh OUTDIR [--version X.Y.Z] [--seq N] [--expires-days D] [--changelog-file FILE] [--work DIR]"
out=$1
shift
version="" seq=1 expires_days=180 changelog_file="" work="" allow_newer=0
while [ $# -gt 0 ]; do
    case $1 in
    --version) version=${2:?}; shift 2 ;;
    --seq) seq=${2:?}; shift 2 ;;
    --expires-days) expires_days=${2:?}; shift 2 ;;
    --changelog-file) changelog_file=${2:?}; shift 2 ;;
    --work) work=${2:?}; shift 2 ;;
    --allow-newer-deps) allow_newer=1; shift ;;
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
if [ -n "$changelog_file" ]; then
    notes=$(python3 -I "$repo/scripts/release_notes.py" "$version" --file "$changelog_file") || exit 1
else
    notes=$(git show HEAD:CHANGELOG.md | python3 -I "$repo/scripts/release_notes.py" "$version" --file -) || exit 1
fi
SOURCE_DATE_EPOCH=$(git log -1 --format=%ct HEAD)
export SOURCE_DATE_EPOCH

for tool in cmake ninja xcrun codesign otool install_name_tool shasum; do
    command -v "$tool" >/dev/null || die "$tool is required but not installed"
done
brew_prefix=$(brew --prefix 2>/dev/null || echo /opt/homebrew)
libomp=$brew_prefix/opt/libomp/lib/libomp.dylib
openssl_root=$brew_prefix/opt/openssl@3
[ -f "$libomp" ] || die "$libomp missing: brew install libomp"
[ -f "$openssl_root/lib/libssl.a" ] || die "$openssl_root/lib/libssl.a missing: brew install openssl@3"

# macOS 15 is what the Metal kernels (MSL 3.2) need and what the release promises (docs/use/macos.md). libomp ships next to
# the server and OpenSSL is linked in, both from Homebrew, whose bottles are built for the Mac they are installed on: on
# a newer Mac they need that newer macOS (a dry run on macOS 27 got libomp for 26.0 and 1887 linker warnings for
# OpenSSL), and the server would not load on macOS 15
min_macos=15.0
built_for() { # the newest macOS any object of FILE was built for
    otool -l "$1" 2>/dev/null | awk '/LC_BUILD_VERSION/ {f = 1} f && $1 == "minos" {print $2; f = 0}' |
        sort -t. -k1,1n -k2,2n | tail -1
}
for dep in "$libomp" "$openssl_root/lib/libssl.a" "$openssl_root/lib/libcrypto.a"; do
    v=$(built_for "$dep")
    [ -n "$v" ] || die "cannot read the macOS version $dep was built for"
    if [ "${v%%.*}" -gt "${min_macos%%.*}" ]; then
        [ "$allow_newer" = 1 ] || die "$dep is built for macOS $v, newer than $min_macos: build on macOS $min_macos (the CI runner), or --allow-newer-deps for a dry run that is never released"
        printf 'release-build-macos: WARNING %s is built for macOS %s; these files do not run on macOS %s\n' "$dep" "$v" "$min_macos" >&2
    fi
done

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

jobs=${TONA_JOBS:-$(sysctl -n hw.ncpu)}
# No absolute paths in the programs; macOS 15 is what the Metal kernels (MSL 3.2) need (docs/use/macos.md)
flags="-ffile-prefix-map=$src=. -ffile-prefix-map=$work=build -Wdate-time"
common=(-G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_OSX_DEPLOYMENT_TARGET=$min_macos" "-DCMAKE_CXX_FLAGS=$flags"
    "-DCMAKE_OBJCXX_FLAGS=$flags")

echo "== Engine (Metal backend, -j$jobs)"
cmake -S "$src/engine" -B "$engine_build" "${common[@]}" -DTONA_BACKEND_METAL=ON -DTONA_MODEL=/nonexistent/model.gguf \
    -DTONA_LLAMACPP_BUILD=/nonexistent -DOPENSSL_ROOT_DIR="$openssl_root" -DOPENSSL_USE_STATIC_LIBS=TRUE >/dev/null
cmake --build "$engine_build" -j"$jobs" --target tona-server tona tona_metallib

echo "== Assets into $out"
p=macos-arm64
declare -a specs=()
add_file() { # SOURCE ASSET PATH MODE
    install -m "$4" "$1" "$out/$2"
    specs+=("--file" "$2:$3:$4:macos-arm64")
}
add_file "$engine_build/tona-server" $p-tona-server bin/tona-server 0755
add_file "$engine_build/tona.metallib" $p-tona.metallib bin/tona.metallib 0644
add_file "$engine_build/tona" $p-tona bin/tona 0755
add_file "$libomp" $p-libomp.dylib bin/libomp.dylib 0644

# The server finds libomp next to itself. The install name is rewritten, then both files are signed again: an arm64
# program without a valid (ad-hoc) signature is killed by the kernel, and install_name_tool invalidates it.
omp_ref=$(otool -L "$out/$p-tona-server" | awk '/libomp\.dylib/ {print $1}')
[ -n "$omp_ref" ] || die "tona-server does not link libomp.dylib; the OpenMP flags of macos_openmp.cmake changed?"
chmod u+w "$out/$p-tona-server" "$out/$p-libomp.dylib"
install_name_tool -id @loader_path/libomp.dylib "$out/$p-libomp.dylib"
install_name_tool -change "$omp_ref" @loader_path/libomp.dylib "$out/$p-tona-server"
chmod 0755 "$out/$p-tona-server"
chmod 0644 "$out/$p-libomp.dylib"
codesign --force --sign - "$out/$p-libomp.dylib" "$out/$p-tona-server"

# A program may depend on the system only: /usr/lib, /System, and what this release ships next to it. A Homebrew path
# would start on this Mac and fail on every other (the equivalent of scripts/release_needed.py)
for f in tona-server tona libomp.dylib; do
    refs=$(otool -L "$out/$p-$f" | tail -n +2 | awk '{print $1}' | grep -v -e '^/usr/lib/' -e '^/System/' -e '^@loader_path/' || true)
    [ -z "$refs" ] || die "$f depends on files outside the system: $refs"
done
codesign --verify "$out/$p-tona-server" "$out/$p-libomp.dylib" "$out/$p-tona"

own=$p-tona-LICENSES.txt
{
    printf '==== LICENSE (Tona) ====\n\n'
    cat "$src/LICENSE"
    for f in "$src"/engine/third_party/LICENSE-*; do
        printf '\n==== %s ====\n\n' "${f#"$src"/}"
        cat "$f"
    done
    printf '\n==== libomp.dylib (LLVM OpenMP runtime, from Homebrew libomp): licenses/llvm-LICENSE.TXT ====\n\n'
    cat "$src/licenses/llvm-LICENSE.TXT"
    printf '\n==== OpenSSL 3 (statically linked into tona): licenses/openssl-LICENSE.txt ====\n\n'
    cat "$src/licenses/openssl-LICENSE.txt"
} >"$out/$own"
chmod 0644 "$out/$own"
specs+=("--file" "$own:bin/tona-LICENSES.txt:0644:macos-arm64")

echo "== Manifest part and SHA256SUMS"
# The ROCm lock hash is a header field of every manifest; the Mac build has no toolchain but reads the same file
lock_sha=$(shasum -a 256 "$src/toolchain/rocm-10.1.lock" | cut -d' ' -f1)
python3 -I "$repo/scripts/release_manifest.py" create --dir "$out" --output "$out/manifest.json" \
    --version "$version" --seq "$seq" --expires-days "$expires_days" --source-date-epoch "$SOURCE_DATE_EPOCH" \
    --commit "$commit" --lock-sha256 "$lock_sha" --changelog "$notes" "${specs[@]}"
# computed before the file exists, so that it does not list itself (as release-merge.sh does)
sums=$(cd "$out" && find . -maxdepth 1 -type f ! -name manifest.json -exec basename {} \; | sort | xargs shasum -a 256)
printf '%s\n' "$sums" >"$out/SHA256SUMS"
echo "done: $(find "$out" -maxdepth 1 -type f | wc -l | tr -d ' ') files, version $version, commit $commit"
