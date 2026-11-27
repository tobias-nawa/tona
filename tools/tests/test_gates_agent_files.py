import agent_files
from conftest import write

TEMPLATE = "tools/agent_files/optimize-for-hardware.md"
TEMPLATE2 = "tools/agent_files/research-before-building.md"
COPIES = [f"{d}/optimize-for-hardware/SKILL.md" for d in (".agents/skills", ".claude/skills", ".github/skills")]


def write_templates(repo, text):
    write(repo, TEMPLATE, text)
    write(repo, TEMPLATE2, text)


def run(*args):
    return agent_files.main(list(args))


def test_generates_three_identical_copies(repo):
    write_templates(repo, "skill text\n")
    assert run() == 0
    assert all((repo / c).read_text() == "skill text\n" for c in COPIES)
    assert run("--check") == 0


def test_second_skill_is_generated_next_to_the_first(repo):
    write_templates(repo, "x\n")
    run()
    assert all(
        (repo / f"{d}/research-before-building/SKILL.md").is_file()
        for d in (".agents/skills", ".claude/skills", ".github/skills")
    )


def test_check_reports_missing_and_changed_copy(repo, capsys):
    write_templates(repo, "skill text\n")
    run()
    (repo / COPIES[0]).unlink()
    write(repo, COPIES[1], "edited\n")
    assert run("--check") == 1
    err = capsys.readouterr().err
    assert COPIES[0] in err and COPIES[1] in err and COPIES[2] not in err
    assert not (repo / COPIES[0]).exists()  # --check changes nothing


def test_template_change_makes_check_fail_until_regenerated(repo):
    write_templates(repo, "v1\n")
    run()
    write(repo, TEMPLATE2, "v2\n")
    assert run("--check") == 1
    run()
    assert run("--check") == 0


def test_nested_agents_md_gets_claude_import(repo, capsys):
    write_templates(repo, "x\n")
    write(repo, "engine/backend/foo/AGENTS.md", "rules\n")
    assert run("--check") == 1
    assert "engine/backend/foo/CLAUDE.md" in capsys.readouterr().err
    run()
    assert (repo / "engine/backend/foo/CLAUDE.md").read_text() == "@AGENTS.md\n"
    assert run("--check") == 0


def test_root_and_vendored_agents_md_get_no_claude_file(repo):
    write_templates(repo, "x\n")
    write(repo, "AGENTS.md", "root\n")
    write(repo, "engine/third_party/lib/AGENTS.md", "vendored\n")
    run()
    assert not (repo / "CLAUDE.md").exists()
    assert not (repo / "engine/third_party/lib/CLAUDE.md").exists()


def test_root_gets_the_gemini_import_only_with_an_agents_md(repo):
    write_templates(repo, "x\n")
    run()
    assert not (repo / "GEMINI.md").exists()
    write(repo, "AGENTS.md", "root\n")
    assert run("--check") == 1
    run()
    assert (repo / "GEMINI.md").read_text() == "@AGENTS.md\n"
    assert run("--check") == 0
