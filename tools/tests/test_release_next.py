"""scripts/release-next.py: the next version and the CHANGELOG draft come from the subjects since the last tag."""

from conftest import commit_all, git, load_script, write

rn = load_script("scripts/release-next.py")


def tagged(repo, tag="v0.2.3"):
    write(repo, "a", "a")
    commit_all(repo, "chore: start")
    git(repo, "tag", tag)


def add(repo, subject, body=""):
    write(repo, "f", subject)
    commit_all(repo, subject + (f"\n\n{body}" if body else ""))


def run(repo, capsys, *extra):
    code = rn.main(["--repo", str(repo), "--date", "2026-12-01", *extra])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_a_fix_is_a_patch(repo, capsys):
    tagged(repo)
    add(repo, "fix(server): stop the stall (#4)")
    code, out, _ = run(repo, capsys)
    assert code == 0 and "git tag -s v0.2.4" in out
    assert "## [0.2.4] - 2026-12-01\n\n### Fixed\n\n- server: stop the stall (#4)\n" in out


def test_a_feature_in_0x_is_a_minor_and_the_highest_change_wins(repo, capsys):
    tagged(repo)
    add(repo, "fix: a (#1)")
    add(repo, "feat(client): add a tab (#2)")
    code, out, _ = run(repo, capsys)
    assert code == 0 and "v0.3.0" in out and "### Added" in out and "### Fixed" in out


def test_a_breaking_change_in_0x_is_a_minor(repo, capsys):
    tagged(repo)
    add(repo, "refactor(installer)!: rename the setting (#9)")
    code, out, _ = run(repo, capsys)
    assert code == 0 and "v0.3.0" in out and "- **Breaking:** installer: rename the setting (#9)" in out


def test_a_breaking_change_paragraph_counts(repo, capsys):
    tagged(repo)
    add(repo, "fix(server): move the port (#9)", "BREAKING CHANGE: the port is 8662 now.")
    code, out, _ = run(repo, capsys)
    assert code == 0 and "v0.3.0" in out


def test_from_1_0_a_breaking_change_is_a_major(repo, capsys):
    tagged(repo, "v1.4.2")
    add(repo, "fix!: change the api (#9)")
    assert "v2.0.0" in run(repo, capsys)[1]


def test_from_1_0_a_feature_is_a_minor_and_a_fix_a_patch(repo, capsys):
    tagged(repo, "v1.4.2")
    add(repo, "feat: x (#1)")
    assert "v1.5.0" in run(repo, capsys)[1]


def test_security_fixes_get_their_own_section(repo, capsys):
    tagged(repo)
    add(repo, "fix(update): reject a manifest with a bad security field (#5)")
    assert "### Security" in run(repo, capsys)[1]


def test_only_housekeeping_proposes_no_release(repo, capsys):
    tagged(repo)
    add(repo, "docs: fix a link")
    add(repo, "ci(deps): bump the actions group")
    code, out, _ = run(repo, capsys)
    assert code == rn.NOTHING and "no release proposed" in out


def test_non_conforming_subjects_are_listed_as_errors(repo, capsys):
    tagged(repo)
    add(repo, "Fix the stall")
    add(repo, "feat(gates): x (#1)")
    code, out, err = run(repo, capsys)
    assert code == 1 and out == ""
    assert "'Fix the stall'" in err and "scope `gates`" in err


def test_only_commits_since_the_tag_count(repo, capsys):
    write(repo, "a", "a")
    commit_all(repo, "Old free-form subject")
    git(repo, "tag", "v0.1.0")
    add(repo, "fix: b (#2)")
    assert run(repo, capsys)[0] == 0


def test_without_a_tag_the_first_release_is_0_1_0(repo, capsys):
    add(repo, "Anything")
    code, out, _ = run(repo, capsys)
    assert code == 0 and "first release 0.1.0" in out and "git tag -s v0.1.0" in out


def test_write_inserts_the_section_below_unreleased(repo, capsys):
    tagged(repo)
    write(repo, "CHANGELOG.md", "# Changelog\n\n## [Unreleased]\n\n## [0.2.3] - 2026-11-01\n\n- old\n")
    add(repo, "fix: b (#2)")
    run(repo, capsys, "--write")
    text = (repo / "CHANGELOG.md").read_text()
    assert text.index("## [Unreleased]") < text.index("## [0.2.4] - 2026-12-01") < text.index("## [0.2.3]")
