"""Exercises of the Aider polyglot benchmark (Exercism practice exercises): discovery, prompt, reply parsing.

Source: the Aider-AI/polyglot-benchmark repository (MIT, Exercism content CC BY-SA 4.0 / MIT per exercise), unpacked
by hand to a directory outside the repository; this module only reads it.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

LANGS = ("python", "go", "rust", "javascript")
# Added 2026-11-19. Not in the default LANGS: the shuffled order of the existing languages must not change.
NEW_LANGS = ("java", "cpp", "php", "typescript", "dart", "bash")
# Build and package files the model never edits, even when an exercise lists them as a solution file (Aider ignores
# Cargo.toml and CMakeLists.txt; the others are the manifests of the new languages). Test files are protected as well.
PROTECTED = frozenset(
    {"Cargo.toml", "CMakeLists.txt", "build.gradle", "package.json", "pubspec.yaml", "composer.json", "tsconfig.json"}
)

PROMPT = """{instructions}

Use the instructions above to modify the supplied file(s): {names}
Do not change the names of existing functions or classes, they are referenced by unit tests that you cannot see.
Only use the standard library; do not suggest installing packages.

Reply with the complete new content of every file you change, one after the other, each in this form:

FILE: <name>
```
<complete file content>
```

Current content of the file(s):
{files}"""

RETRY = """The tests failed. Test output (the end of it):

```
{output}
```

Fix the code so that all tests pass. Reply again with the complete content of every file you change, in the same
form (a line `FILE: <name>`, then a fenced block)."""

FENCE = re.compile(r"^[ \t]*FILE:[ \t]*`?([^\s`]+)`?[ \t]*\n[ \t]*```[^\n]*\n(.*?)\n[ \t]*```[ \t]*$", re.S | re.M)
FALLBACK = re.compile(r"^[ \t]*```[^\n]*\n(.*?)\n[ \t]*```[ \t]*$", re.S | re.M)
THINK = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class Exercise:
    lang: str
    name: str
    root: Path
    solution: list[str]
    tests: list[str]
    example: str
    extra: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.lang}/{self.name}"


def load(base: Path, lang: str) -> list[Exercise]:
    out = []
    for d in sorted((base / lang / "exercises" / "practice").iterdir()):
        meta = json.loads((d / ".meta" / "config.json").read_text())["files"]
        solution = [s for s in meta["solution"] if s not in PROTECTED and s not in meta["test"]]
        out.append(Exercise(lang, d.name, d, solution, meta["test"], meta["example"][0], {"examples": meta["example"]}))
    return out


def reference_files(ex: Exercise) -> dict[str, str]:
    """Files that turn the stub into the exercise's reference solution (name in the staged copy -> content).

    The existing languages have one example file for the first solution file. The new ones may have several: Java puts
    every example class next to the solution class, C++ pairs example.cpp / example.h with the solution file of the same
    suffix, PHP, TypeScript and Dart have one of each. A single example may stand for several stub files (PHP
    word-search: one file holds all three classes): it goes into the solution file named like the exercise and the other
    stubs are emptied, else the classes would be declared twice."""
    if ex.lang not in NEW_LANGS:
        return {ex.solution[0]: (ex.root / ex.example).read_text()}
    out, free = {}, list(ex.solution)
    for e in ex.extra["examples"]:
        suffix = Path(e).suffix
        match = [s for s in free if Path(s).suffix == suffix]
        match.sort(key=lambda s: Path(s).stem.lower().replace("_", "") != ex.name.replace("-", ""))
        if ex.lang != "java" and match:
            dest = match[0]
            free.remove(dest)
        else:
            dest = str(Path(ex.solution[0]).parent / Path(e).name)
        out[dest] = (ex.root / e).read_text()
    if ex.lang != "java":
        out.update({s: "<?php\n" if ex.lang == "php" else "" for s in free})
    return out


def instructions(ex: Exercise) -> str:
    docs = ex.root / ".docs"
    parts = [
        (docs / n).read_text() for n in ("instructions.md", "instructions.append.md", "hints.md") if (docs / n).exists()
    ]
    return "\n\n".join(p.strip() for p in parts)


def first_prompt(ex: Exercise) -> str:
    files = "\n\n".join(f"{n}\n```\n{(ex.root / n).read_text()}\n```" for n in ex.solution)
    return PROMPT.format(instructions=instructions(ex), names=", ".join(ex.solution), files=files)


def parse_reply(text: str, allowed: list[str]) -> dict[str, str]:
    """Files named in the reply (last block per name wins); names outside the editable set are ignored."""
    text = THINK.sub("", text)
    found = {}
    for name, body in FENCE.findall(text):
        match = [a for a in allowed if a == name or a.endswith("/" + name) or name.endswith("/" + a)]
        if match:
            found[match[0]] = body + "\n"
    if not found and len(allowed) == 1:
        # No FILE line, but the exercise has one editable file: take the longest fenced block (the full file, not a
        # usage snippet). A model that skips the header still wrote the code; counting it as "no file" would measure
        # format habits, not the language (2026-11-19, Rust/Go run: 41 of 44 header-less Coder replies hold a block).
        blocks = FALLBACK.findall(text)
        if blocks:
            found[allowed[0]] = max(blocks, key=len) + "\n"
    return found
