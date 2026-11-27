import sys

import pytest
import workflows
from conftest import write

GOOD = """name: ok
on:
  pull_request:
permissions: {}
jobs:
  a:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - env:
          TITLE: ${{ github.event.pull_request.title }}
        run: |
          echo "$TITLE"
"""


def run(monkeypatch, *files):
    monkeypatch.setattr(sys, "argv", ["workflows.py", *files])
    return workflows.main()


def with_job(extra: str) -> str:
    return f"name: t\non: push\njobs:\n  a:\n{extra}\n"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (with_job("    permissions:\n      statuses: write\n    runs-on: ubuntu-latest"), "statuses: write"),
        ("name: t\non: push\npermissions: write-all\njobs: {}\n", "statuses: write"),
        (with_job("    runs-on: self-hosted"), "self-hosted"),
        (with_job("    runs-on: [self-hosted, linux]"), "self-hosted"),
        (
            with_job(
                "    runs-on: ubuntu-latest\n    steps:\n      - run: echo ${{ github.event.head_commit.message }}"
            ),
            "github.event",
        ),
        (
            with_job(
                "    runs-on: ubuntu-latest\n    steps:\n      - run: |\n          echo hi\n"
                "          echo '${{ github.event.pull_request.title }}'"
            ),
            "github.event",
        ),
    ],
)
def test_each_forbidden_pattern_is_rejected(repo, monkeypatch, capsys, text, message):
    write(repo, ".github/workflows/w.yml", text)
    assert run(monkeypatch, ".github/workflows/w.yml") == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "step",
    [
        "      - uses: anthropics/claude-code-action@v1",
        "      - uses: anthropics/claude-code-security-review@abc",
        "      - uses: google-github-actions/run-gemini-cli@v0",
        "      - uses: github/copilot-setup@v1",
        "      - uses: openai/codex-action@v1",
        "      - uses: actions/ai-inference@v1",
        "      - uses: coderabbitai/ai-pr-reviewer@latest",
        "      - run: curl https://api.anthropic.com/v1/messages",
        "      - run: npm install -g @anthropic-ai/claude-code",
        "      - run: pip install openai",
        "      - env:\n          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}\n        run: true",
        "      - run: curl https://models.github.ai/inference/chat",
    ],
)
def test_llm_services_are_rejected(repo, monkeypatch, capsys, step):
    write(repo, ".github/workflows/w.yml", with_job(f"    runs-on: ubuntu-latest\n    steps:\n{step}"))
    assert run(monkeypatch, ".github/workflows/w.yml") == 1
    assert "LLM service" in capsys.readouterr().err


def test_ordinary_actions_and_commented_llm_mentions_pass(repo, monkeypatch):
    steps = (
        "      - uses: actions/checkout@abc\n"
        "      # anthropics/claude-code-action is not used\n"
        "      - run: pip install pytest\n"
    )
    write(repo, ".github/workflows/w.yml", with_job(f"    runs-on: ubuntu-latest\n    steps:\n{steps}"))
    assert run(monkeypatch, ".github/workflows/w.yml") == 0


def test_allowed_workflow_passes_and_env_indirection_is_fine(repo, monkeypatch):
    write(repo, ".github/workflows/w.yml", GOOD)
    assert run(monkeypatch, ".github/workflows/w.yml") == 0


def test_mentions_in_comments_and_outside_run_do_not_count(repo, monkeypatch):
    text = GOOD + "      # a self-hosted runner and statuses: write would be refused\n"
    text += "      - name: ${{ github.event.pull_request.number }}\n        run: true\n"
    write(repo, ".github/workflows/w.yml", text)
    assert run(monkeypatch, ".github/workflows/w.yml") == 0


def test_files_outside_the_workflow_directory_are_ignored(repo, monkeypatch):
    write(repo, "other/w.yml", "runs-on: self-hosted\n")
    assert run(monkeypatch, "other/w.yml") == 0


def test_without_arguments_checks_the_tracked_workflows(repo, monkeypatch):
    write(repo, ".github/workflows/w.yml", with_job("    runs-on: self-hosted"))
    assert run(monkeypatch) == 1


TARGET = "name: t\non:\n  pull_request_target:\njobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n{step}\n"


@pytest.mark.parametrize(
    "step",
    [
        "      - uses: actions/checkout@abc\n        with:\n          ref: ${{ github.event.pull_request.head.sha }}",
        "      - uses: actions/checkout@abc\n        with:\n          ref: ${{ github.head_ref }}",
        "      - uses: actions/checkout@abc\n        with:\n          ref: refs/pull/1/merge",
    ],
)
def test_pull_request_target_must_not_touch_the_head(repo, monkeypatch, capsys, step):
    write(repo, ".github/workflows/w.yml", TARGET.format(step=step))
    assert run(monkeypatch, ".github/workflows/w.yml") == 1
    assert "head of the pull request" in capsys.readouterr().err


def test_pull_request_target_on_the_base_and_head_outside_it_pass(repo, monkeypatch):
    base = "      - uses: actions/checkout@abc\n        with:\n          ref: main  # not pull_request.head\n"
    write(repo, ".github/workflows/w.yml", TARGET.format(step=base))
    write(repo, ".github/workflows/x.yml", GOOD.replace("echo", "echo $X # pull_request.head.sha"))
    other = "name: t\non: pull_request\njobs:\n  a:\n    steps:\n      - with:\n          ref: ${{ github.head_ref }}\n"
    write(repo, ".github/workflows/y.yml", other)
    assert run(monkeypatch, *(f".github/workflows/{n}.yml" for n in "wxy")) == 0
