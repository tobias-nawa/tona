#!/bin/bash
# Switches on the GitHub repository settings that exist only as API settings (rulesets, security features, Actions
# policy). Everything that can live in files is under .github/. See docs/contribute/github.md.
#
# Usage: scripts/github-setup.sh OWNER/REPO [--apply]
# Without --apply it prints the calls and changes nothing (the default). With --apply it needs `gh auth login` as an
# admin of the repository. Run it again at any time: every call sets a state, none toggles, and the rulesets are
# found by name and updated instead of created twice.
#
# Only settings that are free for public repositories are used.
#   ENFORCEMENT          active (default) or evaluate: the rulesets only record what they would block
#   REQUIRED_APPROVALS   approving reviews needed on a PR (default 0; set 1 once there is a second maintainer,
#                        because nobody can approve their own PR)
set -euo pipefail

# Checks the ruleset of `main` requires: the summary job of each workflow (`ok` jobs: they pass when every job that ran
# passed, so a pull request that skips the heavy jobs is not blocked; scripts/ci-changes.sh says which run).
# "gpu-golden" is a commit status that the maintainers post by hand after
# scripts/golden.sh (no runner has the GPU). The code scanning result of CodeQL is required through its own rule.
# `engine-macos` (tests.yml) is not in the list on purpose: it has not run on GitHub yet, and a required check that
# fails for a runner reason (Homebrew, the Metal toolchain download) would block every pull request. Add it after a
# week of green runs on main.
REQUIRED_CHECKS=(check-ok tests-ok gpu-golden)
REQUIRED_APPROVALS=${REQUIRED_APPROVALS:-0}
# "active" enforces; "evaluate" only records what the rules would have blocked (first day, docs/contribute/github.md)
ENFORCEMENT=${ENFORCEMENT:-active}
# Role id of "Repository admin" in bypass lists. Not documented in the REST reference; check it by hand after the
# first run (docs/contribute/github.md, "Check by hand")
ADMIN_ROLE_ID=5
API_VERSION=2026-03-10

usage() {
    echo "usage: $0 OWNER/REPO [--apply]" >&2
    exit 2
}

[[ $# -ge 1 && $# -le 2 && "$1" == */* ]] || usage
repo=$1
apply=0
if [[ $# -eq 2 ]]; then
    [[ "$2" == --apply ]] || usage
    apply=1
fi
[[ "$REQUIRED_APPROVALS" =~ ^[0-9]+$ ]] || { echo "REQUIRED_APPROVALS must be a number" >&2; exit 2; }
[[ "$ENFORCEMENT" == active || "$ENFORCEMENT" == evaluate ]] || { echo "ENFORCEMENT must be active or evaluate" >&2; exit 2; }

if ((apply)); then
    command -v gh >/dev/null || { echo "gh is required" >&2; exit 1; }
    command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }
else
    echo "# Dry run for $repo: nothing is sent. Add --apply to run these calls."
fi

# call METHOD ROUTE [JSON]: sends the request, or prints it in the dry run
call() {
    local method=$1 route=$2 body=${3:-}
    local path="repos/$repo"
    [[ -z "$route" ]] || path+="/$route"
    local args=(api -X "$method" "$path" -H "X-GitHub-Api-Version:$API_VERSION")
    if ((!apply)); then
        printf 'gh'
        printf ' %q' "${args[@]}"
        if [[ -n "$body" ]]; then
            printf " --input - <<'JSON'\n"
            jq . <<<"$body"
            echo JSON
        else
            echo
        fi
        return
    fi
    if [[ -n "$body" ]]; then
        gh "${args[@]}" --input - <<<"$body" >/dev/null
    else
        gh "${args[@]}" >/dev/null
    fi
}

section() { echo; echo "# $*"; }

# upsert_ruleset NAME JSON: creates the ruleset, or updates the one with that name
upsert_ruleset() {
    local name=$1 body=$2 id=""
    if ((apply)); then
        id=$(gh api "repos/$repo/rulesets" --paginate --jq ".[] | select(.name == \"$name\") | .id")
        if [[ -n "$id" ]]; then
            call PUT "rulesets/$id" "$body"
        else
            call POST rulesets "$body"
        fi
    else
        echo "# POST, or PUT rulesets/<id> if a ruleset named \"$name\" exists"
        call POST rulesets "$body"
    fi
}

# Rulesets: https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset
# Rule types: https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset (request body, "rules")
main_ruleset() {
    local checks
    checks=$(printf '%s\n' "${REQUIRED_CHECKS[@]}" | jq -R '{context: .}' | jq -s .)
    # .github/CODEOWNERS only asks for a review; with approvals required it also makes the owner's review mandatory.
    # With 0 approvals (one maintainer) the maintainer would block their own pull requests.
    local owners=false
    ((REQUIRED_APPROVALS > 0)) && owners=true
    jq -n --arg enforcement "$ENFORCEMENT" --argjson checks "$checks" --argjson approvals "$REQUIRED_APPROVALS" \
        --argjson owners "$owners" '{
        name: "main",
        target: "branch",
        enforcement: $enforcement,
        conditions: {ref_name: {include: ["~DEFAULT_BRANCH"], exclude: []}},
        bypass_actors: [],
        rules: [
            {type: "deletion"},
            {type: "non_fast_forward"},
            {type: "required_linear_history"},
            {type: "pull_request", parameters: {
                required_approving_review_count: $approvals,
                dismiss_stale_reviews_on_push: true,
                require_code_owner_review: $owners,
                require_last_push_approval: false,
                required_review_thread_resolution: true,
                allowed_merge_methods: ["squash"]}},
            {type: "required_status_checks", parameters: {
                strict_required_status_checks_policy: true,
                required_status_checks: $checks}},
            {type: "code_scanning", parameters: {code_scanning_tools: [
                {tool: "CodeQL", alerts_threshold: "errors", security_alerts_threshold: "high_or_higher"}]}}
        ]}'
}

# Release tags: only admins create them, nobody moves or deletes one.
tags_ruleset() {
    jq -n --arg enforcement "$ENFORCEMENT" --argjson admin "$ADMIN_ROLE_ID" '{
        name: "release tags",
        target: "tag",
        enforcement: $enforcement,
        conditions: {ref_name: {include: ["refs/tags/v*"], exclude: []}},
        bypass_actors: [{actor_type: "RepositoryRole", actor_id: $admin, bypass_mode: "always"}],
        rules: [
            {type: "creation"},
            {type: "update", parameters: {update_allows_fetch_and_merge: false}},
            {type: "deletion"},
            {type: "non_fast_forward"}
        ]}'
}

section "Rulesets"
upsert_ruleset main "$(main_ruleset)"
upsert_ruleset "release tags" "$(tags_ruleset)"

section "Merge: squash only, the pull request title and text become the commit message"
echo "# The pull request text carries the rights statement and the AI use line (tools/gates/pr_text.py); a squash"
echo "# merge with these two settings keeps them in the history, which Tona has instead of a DCO or CLA."
call PATCH "" '{"allow_squash_merge": true, "allow_merge_commit": false, "allow_rebase_merge": false, "squash_merge_commit_title": "PR_TITLE", "squash_merge_commit_message": "PR_BODY", "delete_branch_on_merge": true}'

section "Secret scanning and push protection (PATCH repository, security_and_analysis)"
echo "# https://docs.github.com/en/rest/repos/repos#update-a-repository"
call PATCH "" '{"security_and_analysis": {"secret_scanning": {"status": "enabled"}, "secret_scanning_push_protection": {"status": "enabled"}}}'

section "Dependabot alerts and the dependency graph"
echo "# https://docs.github.com/en/rest/repos/repos#enable-vulnerability-alerts"
call PUT vulnerability-alerts

section "Dependabot security updates"
echo "# https://docs.github.com/en/rest/repos/repos#enable-automated-security-fixes"
call PUT automated-security-fixes

section "Private vulnerability reporting"
echo "# https://docs.github.com/en/rest/repos/repos#enable-private-vulnerability-reporting-for-a-repository"
call PUT private-vulnerability-reporting

section "Immutable releases"
echo "# https://docs.github.com/en/rest/repos/repos#enable-immutable-releases"
call PUT immutable-releases

section "Actions: only actions pinned to a full commit SHA"
echo "# https://docs.github.com/en/rest/actions/permissions#set-github-actions-permissions-for-a-repository"
call PUT actions/permissions '{"enabled": true, "allowed_actions": "all", "sha_pinning_required": true}'

section "Actions: the default token reads only, workflows cannot approve PRs"
echo "# https://docs.github.com/en/rest/actions/permissions#set-default-workflow-permissions-for-a-repository"
call PUT actions/permissions/workflow '{"default_workflow_permissions": "read", "can_approve_pull_request_reviews": false}'

section "Actions: a maintainer approves workflow runs of every outside contributor"
echo "# https://docs.github.com/en/rest/actions/permissions#set-the-fork-pr-contributor-approval-permissions-for-a-repository"
call PUT actions/permissions/fork-pr-contributor-approval '{"approval_policy": "all_external_contributors"}'

section "Labels (.github/labels.tsv)"
while IFS=$'\t' read -r name color description; do
    [[ -z $name || $name == \#* ]] && continue
    if ((apply)); then
        gh label create "$name" --repo "$repo" --color "$color" --description "$description" --force >/dev/null
    else
        printf 'gh label create %q --repo %q --color %q --description %q --force\n' "$name" "$repo" "$color" "$description"
    fi
done <"$(dirname "$0")/../.github/labels.tsv"

if ((!apply)); then
    echo
    echo "# End of the dry run. Nothing was sent."
fi
