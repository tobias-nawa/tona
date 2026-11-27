"""Scoring of family A answers (exact match after normalisation) and extraction of family B code."""

import ast
import re
import textwrap

NAME_KINDS = {"base", "caller", "doc"}


def first_line(reply: str) -> str:
    for line in reply.strip().splitlines():
        line = line.strip().strip("`").strip()
        if line and not line.startswith("```"):
            return line
    return ""


def canon_value(s: str) -> str:
    """A Python expression in one canonical spelling (quotes, spacing); other text unchanged."""
    s = s.strip().rstrip(".").strip().strip("`").strip()
    try:
        return ast.unparse(ast.parse(s, mode="eval"))
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return s


def canon_name(s: str) -> str:
    """Last component of a possibly qualified name, without quotes, backticks and call parentheses."""
    s = s.strip().rstrip(".").strip("`'\" ")
    s = re.sub(r"\(.*\)$", "", s)
    s = re.sub(r"\[.*\]$", "", s)  # a base class `Generic[T]` is named `Generic`
    return s.split(".")[-1].strip("`'\" ")


def is_correct(kind: str, expected: str, reply: str) -> bool:
    got = first_line(reply)
    if kind in NAME_KINDS:
        return canon_name(got) == canon_name(expected)
    return canon_value(got) == canon_value(expected)


def code_block(reply: str) -> str:
    """The first fenced code block of the reply (closing fence at the start of a line, so a docstring that mentions
    backticks does not end it), else the reply after an opening fence, else the reply itself."""
    m = re.search(r"```[A-Za-z0-9_+-]*\n(.*?)^```", reply, re.S | re.M)
    if m:
        return m.group(1)
    m = re.search(r"```[A-Za-z0-9_+-]*\n(.*)", reply, re.S)
    return m.group(1) if m else reply


def extract_body(reply: str, name: str, col: int) -> str | None:
    """The statements after the docstring of function `name` in the reply, indented to `col`; None if there is none.

    The reply may hold the whole definition (also inside a class) or only the body.
    """
    code = textwrap.dedent(code_block(reply))  # a method pasted with its class indentation
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _bare_body(code, col)
    if not any(isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) for n in ast.walk(tree)):
        return _bare_body(code, col)  # the reply is the body alone
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name:
            stmts = node.body
            doc = getattr(stmts[0], "value", None)
            if isinstance(stmts[0], ast.Expr) and isinstance(doc, ast.Constant) and isinstance(doc.value, str):
                stmts = stmts[1:]
            if not stmts:
                return None
            lines = code.splitlines()[stmts[0].lineno - 1 : node.end_lineno]
            return _indent(lines, stmts[0].col_offset, col)
    return None


def _bare_body(code: str, col: int) -> str | None:
    lines = code.splitlines()
    nonblank = [ln for ln in lines if ln.strip()]
    if not nonblank:
        return None
    lead = min(len(ln) - len(ln.lstrip()) for ln in nonblank)
    return _indent(lines, lead, col)


def _indent(lines: list, lead: int, col: int) -> str:
    return "".join((" " * col + ln[lead:] if ln.strip() else "") + "\n" for ln in lines)
