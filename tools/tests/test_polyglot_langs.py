"""Plumbing of the five polyglot languages added later (java, cpp, php, typescript, dart): which files the model may
edit, the prompt, the test commands, enabling skipped tests, the reference solution, the slug selection; and that the
five existing languages (python, go, rust, javascript, csharp) are untouched."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench/polyglot"))
import prepare
import sandbox
import tasks

FENCE = "```"


def make_exercise(root: Path, lang: str, slug: str, files: dict, *spec: list) -> tasks.Exercise:
    solution, test, example = spec
    ex_dir = root / lang / "exercises/practice" / slug
    for name, text in files.items():
        (ex_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (ex_dir / name).write_text(text)
    meta = {"files": {"solution": solution, "test": test, "example": example}}
    (ex_dir / ".meta").mkdir(exist_ok=True)
    (ex_dir / ".meta/config.json").write_text(json.dumps(meta))
    (ex_dir / ".docs").mkdir(exist_ok=True)
    (ex_dir / ".docs/instructions.md").write_text("Do the thing.")
    return next(e for e in tasks.load(root, lang) if e.name == slug)


@pytest.fixture
def java(tmp_path):
    files = {
        "build.gradle": "plugins { id 'java' }",
        "src/main/java/Game.java": "class Game {}",
        "src/test/java/GameTest.java": '@Disabled("Remove to run test")\n@Test void a() {}\n@Disabled\nvoid b() {}',
        ".meta/src/reference/java/Game.java": "class Game { int x; }",
        ".meta/src/reference/java/Frame.java": "class Frame {}",
    }
    return make_exercise(
        tmp_path, "java", "game", files, ["src/main/java/Game.java", "build.gradle"],
        ["src/test/java/GameTest.java"], [".meta/src/reference/java/Game.java", ".meta/src/reference/java/Frame.java"],
    )  # fmt: skip


@pytest.fixture
def cpp(tmp_path):
    files = {
        "CMakeLists.txt": "project(x)",
        "clock.cpp": "// stub",
        "clock.h": "// stub",
        "clock_test.cpp": "// tests",
        ".meta/example.cpp": "// ref cpp",
        ".meta/example.h": "// ref h",
    }
    return make_exercise(
        tmp_path, "cpp", "clock", files, ["clock.cpp", "clock.h", "CMakeLists.txt"],
        ["clock_test.cpp"], [".meta/example.cpp", ".meta/example.h"],
    )  # fmt: skip


def test_manifests_and_tests_are_not_editable(java, cpp):
    assert java.solution == ["src/main/java/Game.java"]
    assert cpp.solution == ["clock.cpp", "clock.h"]


def test_prompt_shows_only_editable_files(java):
    prompt = tasks.first_prompt(java)
    assert "src/main/java/Game.java" in prompt and "class Game {}" in prompt
    assert "build.gradle" not in prompt and "GameTest" not in prompt


def test_parse_fallback_with_one_editable_file_and_header_for_several(java, cpp):
    text = f"Here:\n{FENCE}java\nclass Game {{ int y; }}\n{FENCE}"
    assert tasks.parse_reply(text, java.solution) == {"src/main/java/Game.java": "class Game { int y; }\n"}
    assert tasks.parse_reply(text, cpp.solution) == {}  # two editable files: the FILE header is needed
    named = f"FILE: clock.h\n{FENCE}\n// new\n{FENCE}\nFILE: CMakeLists.txt\n{FENCE}\nevil\n{FENCE}"
    assert tasks.parse_reply(named, cpp.solution) == {"clock.h": "// new\n"}  # a protected file is ignored


def test_reference_files(java, cpp):
    assert tasks.reference_files(java) == {
        "src/main/java/Game.java": "class Game { int x; }",
        "src/main/java/Frame.java": "class Frame {}",
    }
    assert tasks.reference_files(cpp) == {"clock.cpp": "// ref cpp", "clock.h": "// ref h"}


def test_disabled_annotations_are_removed_from_java_tests(java, tmp_path):
    work = tmp_path / "w"
    sandbox.stage(java, {"src/main/java/Game.java": "class Game { int y; }"}, work)
    test = (work / "src/test/java/GameTest.java").read_text()
    assert "@Disabled(" not in test and "@Test void a() {}" in test
    assert (work / "src/main/java/Game.java").read_text() == "class Game { int y; }"
    assert not (work / ".meta").exists() and not (work / ".docs").exists()


def test_skipped_tests_are_enabled_for_typescript_dart_php(tmp_path):
    ts = (
        "xit('a', f)\nxdescribe('b', g)\nit.skip('c', h)\nxtest('d', i)",
        "it('a', f)\ndescribe('b', g)\nit('c', h)\ntest('d', i)",
    )
    dart = (
        "test('a', f, skip: true);\ntest('b', g, skip: false);",
        "test('a', f, skip: false);\ntest('b', g, skip: false);",
    )
    php = ("  $this->markTestSkipped('x');\n  $this->assertTrue(true);\n", "  $this->assertTrue(true);\n")
    cases = {"typescript": ("a.test.ts", *ts), "dart": ("a_test.dart", *dart), "php": ("ATest.php", *php)}
    for lang, (name, before, after) in cases.items():
        ex = make_exercise(tmp_path, lang, "a", {name: before, "s": ""}, ["s"], [name], [".meta/e"])
        work = tmp_path / f"w-{lang}"
        sandbox.stage(ex, {}, work)
        assert (work / name).read_text() == after, lang


def capture_command(monkeypatch, tmp_path, ex) -> list[str]:
    seen = []

    def fake_run(cmd, **_):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sandbox, "STAGE_DIR", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert sandbox.run_tests(ex, {})[0]
    return seen[0]


def test_test_commands_of_the_new_languages(monkeypatch, tmp_path, java, cpp):
    cmd = capture_command(monkeypatch, tmp_path, java)
    assert "--network=none" in cmd and "localhost/tona-poly-java" in cmd
    assert cmd[-1] == "gradle --offline --no-daemon test"
    assert "GRADLE_RO_DEP_CACHE=/opt/gradle-home/caches" in cmd and "--workdir=/work" in cmd
    cmd = capture_command(monkeypatch, tmp_path, cpp)
    assert cmd[-1].endswith('cmake -DEXERCISM_RUN_ALL_TESTS=1 -G "Unix Makefiles" .. && make')
    assert "--workdir=/work/clock" in cmd  # CMake names the target after the directory
    assert any(c.endswith(":/work/clock:rw,z") for c in cmd)


def test_bash_runs_bats_with_skipped_tests_enabled(monkeypatch, tmp_path):
    ex = make_exercise(
        tmp_path, "bash", "bob", {"bob.sh": "", "bob.bats": ""}, ["bob.sh"], ["bob.bats"], [".meta/e.sh"]
    )
    cmd = capture_command(monkeypatch, tmp_path, ex)
    assert cmd[-1] == "bats bob.bats" and "BATS_RUN_SKIPPED=true" in cmd


def test_php_command_names_the_test_file(monkeypatch, tmp_path):
    ex = make_exercise(tmp_path, "php", "a", {"A.php": "", "ATest.php": ""}, ["A.php"], ["ATest.php"], [".meta/e"])
    assert capture_command(monkeypatch, tmp_path, ex)[-1] == "phpunit ATest.php"


def test_existing_languages_keep_their_commands_and_prompt():
    """Frozen on 2026-11-19, before the new languages: commands of the five existing languages and the prompt texts."""
    assert sandbox.COMMANDS["python"] == "python -m pytest -q -x -p no:cacheprovider {tests}"
    assert sandbox.COMMANDS["go"] == "go test ./..."
    assert sandbox.COMMANDS["rust"] == "cargo test --offline -- --include-ignored"
    assert sandbox.COMMANDS["javascript"] == sandbox.JS_PREP + "; npx jest --ci ./*"
    assert sandbox.COMMANDS["csharp"] == sandbox.CS_PREP + "; " + sandbox.CS_TEST
    assert sandbox.JS_PREP == (
        r"sed -i 's/\bxtest(/test(/;s/\bxit(/it(/;s/\bxdescribe(/describe(/' *.spec.js; "
        "ln -s /opt/node_modules node_modules"
    )
    assert tasks.LANGS == ("python", "go", "rust", "javascript")
    digest = hashlib.sha256((tasks.PROMPT + "\x00" + tasks.RETRY).encode()).hexdigest()
    assert digest == "d22c522340f399bf06811fa90a8f169d6a0c88393b73be5c7c721519c73759e4"
    assert sandbox.LANG_ENV.keys() >= {"csharp"} and "python" not in sandbox.LANG_ENV


def test_existing_language_reference_is_the_first_example_over_the_first_solution(tmp_path):
    ex = make_exercise(tmp_path, "rust", "a", {"src/lib.rs": "", "Cargo.toml": "", ".meta/example.rs": "ref"},
                       ["src/lib.rs", "Cargo.toml"], ["tests/a.rs"], [".meta/example.rs"])  # fmt: skip
    assert tasks.reference_files(ex) == {"src/lib.rs": "ref"}
    assert ex.solution == ["src/lib.rs"]


def test_slug_selection_follows_aider_and_drops_deprecated(tmp_path):
    aider = tmp_path / "aider"
    for lang, slugs in {"python": ["bowling", "clock"], "go": ["bowling", "zipper"]}.items():
        for s in slugs:
            (aider / lang / "exercises/practice" / s).mkdir(parents=True)
    track = tmp_path / "track"
    for s in ["bowling", "clock", "leap", "zipper"]:
        (track / "exercises/practice" / s).mkdir(parents=True)
    practice = [{"slug": "clock", "status": "deprecated"}, {"slug": "bowling"}]
    (track / "config.json").write_text(json.dumps({"exercises": {"practice": practice}}))
    assert prepare.select(aider, track) == ["bowling", "zipper"]
    prepare.link(track, "php", tmp_path / "out", ["bowling", "zipper"])
    assert (tmp_path / "out/php/exercises/practice/zipper").is_symlink()
    assert (tmp_path / "out/php/selection.txt").read_text() == "bowling\nzipper\n"


def test_one_example_for_several_php_stubs_goes_into_the_main_file(tmp_path):
    files = {
        "Location.php": "<?php class L {}",
        "WordSearch.php": "<?php class W {}",
        "T.php": "",
        ".meta/example.php": "ref",
    }
    solution = ["Location.php", "WordSearch.php"]
    ex = make_exercise(tmp_path, "php", "word-search", files, solution, ["T.php"], [".meta/example.php"])
    assert tasks.reference_files(ex) == {"WordSearch.php": "ref", "Location.php": "<?php\n"}


def test_agent_report_has_a_row_per_new_language_and_their_sum(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench/agent"))
    import report

    for arm, ok in (("a", True), ("b", False)):
        for lang in ("java", "php"):
            rows = [{"task": f"{lang}/t{i}", "ok": ok or i == 0} for i in range(4)]
            (tmp_path / f"{arm}-{lang}-r0.json").write_text(json.dumps(rows))
    text = report.tables(tmp_path, "a", "b", ["java", "php"])
    assert "new languages (sum)" in text and "- java:" in text and "- php:" in text
