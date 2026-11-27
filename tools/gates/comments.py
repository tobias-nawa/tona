"""Comment gate: the mechanical part of the comment rules in AGENTS.md ("Comments").

Usage: comments.py [FILE...]  (no arguments: every tracked C++/HIP/Python/shell file)

Fails, with file:line, on
- `TODO` or `FIXME` without an issue number (`TODO(#48)` is the only accepted form);
- banner comments: a comment that is only a line of `=`, `-`, `*`, `#`, `~` or `_`, or a title between two such runs;
- commented-out code: a comment on a line of its own that parses as a statement (heuristics below).

"Why, not what" and "a number names its source" stay review items (docs/contribute/verification.md, review checklist):
no parser can judge them.

Commented-out code heuristics, chosen to find real dead code and to stay silent on prose:
- Python: the comment text must parse with `ast` as an assignment, a call, an import, a return/raise/del/assert, or a
  compound statement header (`if x:`). A bare name, a literal, `Label: text` and directives (`noqa`, `type:`) do not.
- C++/HIP: a preprocessor line, a statement that ends with `;` and has a call, an assignment or a member access, a
  control-flow header (`if (...)`, `for (...)`, `while (...)`, `} else {`), or a lone closing brace in a run of such
  lines.
- shell: control-flow lines (`if ...; then`, `for ...; do`, `fi`, `done`) and plain variable assignments.
Trailing comments (code before them on the line) are not checked for code, only for TODO.
"""

import ast
import io
import re
import sys
import tokenize
from typing import NamedTuple

from common import CPP_SUFFIXES, fail_if_empty, gate_files, repo_root, report

C_SUFFIXES = (*CPP_SUFFIXES, ".mm", ".metal")
SUFFIXES = (*C_SUFFIXES, ".py", ".sh")

TODO = re.compile(r"\b(TODO|FIXME)\b(?!\(#\d+\))")
# A run of at least three rule characters, alone or as the frame of a title: `// ==== Setup ====`
RULE_CHARS = "=-*#~_"
BANNER = re.compile(
    rf"^[{re.escape(RULE_CHARS)}]{{3,}}$|^[{re.escape(RULE_CHARS)}]{{3,}}\s.*\s[{re.escape(RULE_CHARS)}]{{3,}}$"
)

CPP_PREPROCESSOR = re.compile(r"^#\s*(include\s*[<\"]|define\s+\w|ifdef\s|ifndef\s|if\s+[\w(!]|endif\b|undef\s)")
CPP_CONTROL = re.compile(r"^(\}\s*)?(else\s+)?(if|for|while|switch)\s*\(.*\)\s*\{?$|^\}?\s*else\s*\{?$|^do\s*\{$")
CPP_STATEMENT = re.compile(
    r"^(return\b.*|break|continue|goto\s+\w+|[\w:<>,\*&\s\[\]\.\->~]*[\w\]\)]\s*(=|\+=|-=|\*=|/=|\|=|&=)[^=].*|[\w:\.\->\[\]]+\s*\(.*\)|\+\+\w+|\w+\+\+|--\w+|\w+--)\s*;$"
)
# prose that ends in `;` after a call-looking word, e.g. "see foo(); then bar"
NON_ASCII = re.compile(r"[^\x00-\x7f]")
LABEL = re.compile(r"(?<!:)\b\w+:\s")
PROSE_WORDS = re.compile(r"[A-Za-z]{3,}\s+[A-Za-z]{3,}\s+[A-Za-z]{3,}")

SHELL_CONTROL = re.compile(r"^(if\s.*;\s*then|for\s.*;\s*do|while\s.*;\s*do|fi|done|esac|then|else)$")
SHELL_ASSIGN = re.compile(r"^(export\s+)?[A-Za-z_][A-Za-z0-9_]*=(\$\(|\"|'|\$\{)\S*.*$")


class Comment(NamedTuple):
    line: int
    text: str
    standalone: bool


def c_comments(source: str) -> list[Comment]:
    """Comments of a C-family file: `//` and `/* */`, skipping string, character and raw-string literals."""
    out: list[Comment] = []
    i, n, line = 0, len(source), 1
    line_start = 0
    while i < n:
        ch = source[i]
        two = source[i : i + 2]
        if ch == "\n":
            line += 1
            line_start = i + 1
            i += 1
        elif source.startswith('R"', i) and (m := re.match(r'R"([^(\s]{0,16})\(', source[i:])):
            end = source.find(")" + m.group(1) + '"', i)
            end = n if end < 0 else end + len(m.group(1)) + 2
            line += source.count("\n", i, end)
            i = end
        elif ch in "\"'":
            i = skip_literal(source, i, ch)
        elif two == "//":
            end = source.find("\n", i)
            end = n if end < 0 else end
            # A line continued with a backslash continues the comment too; no gate reads that far
            alone = not source[line_start:i].strip()
            out.append(Comment(line, source[i + 2 : end], alone))
            i = end
        elif two == "/*":
            end = source.find("*/", i + 2)
            end = n if end < 0 else end
            alone = not source[line_start:i].strip()
            for k, part in enumerate(source[i + 2 : end].split("\n")):
                out.append(Comment(line + k, part.lstrip().removeprefix("*").rstrip(), alone))
            line += source.count("\n", i, end)
            i = min(end + 2, n)
        else:
            i += 1
    return out


def skip_literal(source: str, i: int, quote: str) -> int:
    n = len(source)
    i += 1
    while i < n and source[i] != quote and source[i] != "\n":
        i += 2 if source[i] == "\\" else 1
    return i + 1


def py_comments(source: str) -> list[Comment]:
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                alone = not tok.line[: tok.start[1]].strip()
                out.append(Comment(tok.start[0], tok.string[1:], alone))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass  # the syntax gates (ruff) report a file that does not tokenize
    return out


def sh_comments(source: str) -> list[Comment]:
    out = []
    for no, raw in enumerate(source.split("\n"), 1):
        pos = shell_comment_start(raw)
        if pos < 0:
            continue
        text = raw[pos + 1 :]
        if no == 1 and text.startswith("!"):
            continue
        out.append(Comment(no, text, not raw[:pos].strip()))
    return out


def shell_comment_start(line: str) -> int:
    quote = ""
    for k, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (k == 0 or line[k - 1].isspace()):
            return k
    return -1


def is_banner(text: str) -> bool:
    return bool(BANNER.match(text.strip()))


def python_code(text: str) -> bool:
    text = text.strip()
    # "e.g. foo(x)" parses as an attribute access, and is prose
    if not text or text.startswith(("!", "e.g.", "i.e.")):
        return False
    for candidate in (text, text + "\n    pass" if text.endswith(":") else None):
        if candidate is None:
            continue
        try:
            tree = ast.parse(candidate)
        except (SyntaxError, ValueError):
            continue
        return any(statement_is_code(node) for node in tree.body)
    return False


def statement_is_code(node: ast.stmt) -> bool:
    if isinstance(node, ast.Expr):
        return isinstance(node.value, ast.Call)
    if isinstance(node, ast.AnnAssign):
        return node.value is not None
    return isinstance(
        node,
        (
            ast.Assign,
            ast.AugAssign,
            ast.Import,
            ast.ImportFrom,
            ast.Return,
            ast.Raise,
            ast.Delete,
            ast.Assert,
            ast.If,
            ast.For,
            ast.While,
            ast.With,
            ast.Try,
            ast.FunctionDef,
            ast.ClassDef,
        ),
    )


def cpp_code(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if CPP_PREPROCESSOR.match(text) or CPP_CONTROL.match(text):
        return True
    # Prose with a formula: a label (`Per stream c: x = ...;`), a non-ASCII operator, or three plain words in a row
    if NON_ASCII.search(text) or LABEL.search(text) or PROSE_WORDS.search(text):
        return False
    return bool(CPP_STATEMENT.match(text))


def shell_code(text: str) -> bool:
    text = text.strip()
    return bool(SHELL_CONTROL.match(text) or SHELL_ASSIGN.match(text))


def check_file(rel: str, source: str) -> list[str]:
    if rel.endswith(C_SUFFIXES):
        comments, is_code = c_comments(source), cpp_code
    elif rel.endswith(".py"):
        comments, is_code = py_comments(source), python_code
    else:
        comments, is_code = sh_comments(source), shell_code
    problems = []
    for c in comments:
        where = f"{rel}:{c.line}"
        if TODO.search(c.text):
            problems.append(f"{where}: TODO/FIXME without an issue number (write TODO(#n))")
        if is_banner(c.text):
            problems.append(f"{where}: banner comment")
        elif c.standalone and is_code(c.text):
            problems.append(f"{where}: commented-out code: {c.text.strip()[:60]}")
    return problems


def main() -> int:
    root = repo_root()
    files, explicit = gate_files(sys.argv[1:], SUFFIXES)
    if not explicit:
        fail_if_empty(files, "comments")
    problems = []
    for f in files:
        source = (root / f).read_text(encoding="utf-8", errors="replace")
        problems += check_file(f, source)
    hint = "Delete dead code (git keeps it), write TODO(#n) with an issue, drop the banner."
    return report("comments", problems, len(files), hint)


if __name__ == "__main__":
    sys.exit(main())
