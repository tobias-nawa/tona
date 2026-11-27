# shellcheck shell=bash
# Shared by release-sign.sh and release-verify-build.sh (source it).

# The one place that names the GitHub repository for the release scripts; the workflow uses github.repository.
# TONA_RELEASE_REPO overrides it, e.g. for a fork.
TONA_RELEASE_REPO=${TONA_RELEASE_REPO:-tobias-nawa/tona}
# The signing workflow whose attestations are accepted
export TONA_RELEASE_WORKFLOW=$TONA_RELEASE_REPO/.github/workflows/release.yml

die() { printf '%s: %s\n' "$(basename "$0")" "$*" >&2; exit 1; }

need() {
    local tool
    for tool in "$@"; do command -v "$tool" >/dev/null || die "$tool is required but not installed"; done
}

# A tag the release workflow builds: vX.Y.Z
check_tag() {
    [[ $1 =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "tag must look like v1.2.3, got '$1'"
}

# Proof that this machine rebuilt the tag and got the draft's hashes (written by release-verify-build.sh, required by
# release-sign.sh). Per user and outside /tmp; TONA_RELEASE_PROOF_DIR overrides it for tests.
proof_file() {
    printf '%s/%s.proof\n' "${TONA_RELEASE_PROOF_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/tona/release}" "$1"
}
