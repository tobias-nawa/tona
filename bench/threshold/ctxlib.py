"""Context assembly: which files enter the 64K context, where functions sit, how a body is replaced by `...`.

Foreign source is only parsed with ast (never executed or imported).
"""

import ast
from dataclasses import dataclass

import common

HEADER = "### FILE: "


@dataclass
class Func:
    path: str  # file path as shown in the context
    qualname: str  # Class.method or function
    name: str
    start: int  # line of the `def` (1-based, decorators excluded)
    body_start: int  # first line of the body after the docstring
    body_end: int  # last line of the function
    col: int  # indentation of the body in columns
    has_doc: bool

    @property
    def body_lines(self) -> int:
        return self.body_end - self.body_start + 1


def file_text(repo: common.Repo, name: str) -> str:
    return (common.repo_dir(repo) / repo.src_dir / name).read_text()


def cut_top_level(text: str, keep: int) -> str:
    """The first `keep` top-level statements (cutting between statements keeps the text parseable)."""
    nodes = ast.parse(text).body
    if keep >= len(nodes):
        return text
    end = nodes[keep - 1].end_lineno
    return "".join(text.splitlines(keepends=True)[:end])


def plan_files(repo: common.Repo, budget: int = common.CONTEXT_TOKENS) -> list:
    """[(path, text)] in the fixed order, whole files until the token budget; the last file is cut at a statement."""
    files, used = [], 0
    for name in repo.file_order:
        path = f"{repo.src_dir}/{name}"
        text = file_text(repo, name)
        n = common.count_tokens(f"{HEADER}{path}\n{text}\n\n")
        if used + n <= budget:
            files.append((path, text))
            used += n
            continue
        lo, hi = 1, len(ast.parse(text).body)
        while lo < hi:  # largest statement count that still fits
            mid = (lo + hi + 1) // 2
            if used + common.count_tokens(f"{HEADER}{path}\n{cut_top_level(text, mid)}\n\n") <= budget:
                lo = mid
            else:
                hi = mid - 1
        files.append((path, cut_top_level(text, lo)))
        break
    return files


def assemble(files: list) -> tuple:
    """(text, {path: char offset of the file's first line}); same layout for every task of a repo."""
    parts, offsets, pos = [], {}, 0
    for path, text in files:
        head = f"{HEADER}{path}\n"
        offsets[path] = pos + len(head)
        block = f"{head}{text}\n\n"
        parts.append(block)
        pos += len(block)
    return "".join(parts), offsets


def functions_of(path: str, text: str) -> list:
    """Functions and methods that are not nested in another function, with the span of their body."""
    out = []

    def visit(nodes, prefix):
        for n in nodes:
            if isinstance(n, ast.ClassDef):
                visit(n.body, prefix + n.name + ".")
            elif isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef):
                out.append(_func(path, prefix, n))

    visit(ast.parse(text).body, "")
    return out


def _func(path: str, prefix: str, n: ast.FunctionDef) -> Func:
    doc = (
        isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
        and isinstance(n.body[0].value.value, str)
    )
    first = n.body[1] if doc and len(n.body) > 1 else n.body[0]
    # a body that is only a docstring has no statements to replace; mark it empty (body_end < body_start)
    empty = doc and len(n.body) == 1
    return Func(
        path=path,
        qualname=prefix + n.name,
        name=n.name,
        start=n.lineno,
        body_start=n.end_lineno + 1 if empty else first.lineno,
        body_end=n.end_lineno,
        col=n.body[0].col_offset,
        has_doc=doc,
    )


def hole_text(text: str, f: Func) -> str:
    """The file text with the body of f (after the docstring) replaced by `...` (signature and docstring stay)."""
    lines = text.splitlines(keepends=True)
    new = " " * f.col + "...\n"
    return "".join(lines[: f.body_start - 1]) + new + "".join(lines[f.body_end :])


def original_body(text: str, f: Func) -> str:
    return "".join(text.splitlines(keepends=True)[f.body_start - 1 : f.body_end])


def apply_holes(text: str, funcs: list) -> tuple:
    """Replace the bodies of funcs (all in this file) by `...`; returns (new text, {qualname: line of the `...`})."""
    lines, shift, where = text.splitlines(keepends=True), 0, {}
    for f in sorted(funcs, key=lambda f: f.body_start):
        start = f.body_start - 1 - shift
        lines[start : f.body_end - shift] = [" " * f.col + "...\n"]
        where[f.qualname] = start + 1
        shift += f.body_lines - 1
    return "".join(lines), where


def holed_files(plan: list, funcs: list) -> tuple:
    """(files with holes, {(path, qualname): line of the `...` in the holed file})."""
    out, where = [], {}
    for path, text in plan:
        mine = [f for f in funcs if f.path == path]
        new, w = apply_holes(text, mine)
        out.append((path, new))
        where.update({(path, q): ln for q, ln in w.items()})
    return out, where


def line_token_pos(files: list, offsets: dict, text: str, path: str, line: int) -> int:
    """Tokens before the given line of a file in the assembled context."""
    src = dict(files)[path].splitlines(keepends=True)[: line - 1]
    return common.count_tokens(text[: offsets[path] + len("".join(src))])
