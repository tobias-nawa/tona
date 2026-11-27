#!/usr/bin/python3 -I
"""Family A: retrieval questions with exact answers, generated from the AST of the context (nothing is executed).

Usage: gen_a.py REPO COUNT [LOCAL_SHARE]   -> WORK/tasks/REPO-A.jsonl
Kinds: default (default value of a parameter), constant (module-level literal), base (base class), caller (the one
function that calls X), doc (function by the start of its docstring), returns (return annotation), import (module
of an imported name), owner (class of a method with a unique name).
Every question has its source at a known token position. Questions about the long range have their source at least
2,400 tokens before the end of the context (QSA selects nothing below 2,051 positions); about 10 % are local controls
with the source in the last 2,000 tokens, which sharing indexers must not affect.
"""

import ast
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import ctxlib

SEED = 76
MARGIN = 2400
LOCAL_SPAN = 2000
MAX_ANSWER = 60  # characters; longer values are not a fair exact-match target
TRIVIAL = {"None", "True", "False"}
ASK = "Answer with the exact value or name only, on one line, without explanation."


def trivial_ok(answer: str) -> bool:
    return 0 < len(answer) <= MAX_ANSWER and "\n" not in answer


def q_defaults(path, tree, text) -> list:
    out = []
    for f in ctxlib.functions_of(path, text):
        node = _node_at(tree, f)
        a = node.args
        pairs = list(zip(a.args[len(a.args) - len(a.defaults) :], a.defaults, strict=True)) + [
            (k, d) for k, d in zip(a.kwonlyargs, a.kw_defaults, strict=True) if d is not None
        ]
        for arg, d in pairs:
            ans = ast.unparse(d)
            if isinstance(d, ast.Constant | ast.Tuple | ast.UnaryOp) and trivial_ok(ans):
                q = f'In {path}, what is the default value of the parameter "{arg.arg}" of the function "{f.qualname}"?'
                out.append(_item("default", q, ans, d.lineno, f.qualname, trivial=ans in TRIVIAL))
    return out


def _node_at(tree, f: ctxlib.Func):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.lineno == f.start and n.name == f.name:
            return n


def q_constants(path, tree, text) -> list:
    out = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            ans = ast.unparse(n.value)
            if isinstance(n.value, ast.Constant | ast.Tuple | ast.Call | ast.Dict) and trivial_ok(ans):
                name = n.targets[0].id
                if not name.startswith("__"):
                    q = f'In {path}, what is the value assigned to the module-level name "{name}"?'
                    out.append(_item("constant", q, ans, n.lineno, name))
    return out


def q_bases(path, tree, text) -> list:
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.bases and ast.unparse(n.bases[0]) != "object":
            ans = ast.unparse(n.bases[0])
            if trivial_ok(ans):
                q = f'What is the first base class of the class "{n.name}" defined in {path}?'
                out.append(_item("base", q, ans, n.lineno, n.name))
    return out


def q_docs(path, tree, text) -> list:
    out = []
    for f in ctxlib.functions_of(path, text):
        doc = ast.get_docstring(_node_at(tree, f))
        words = (doc or "").split()
        if len(words) >= 10 and not f.name.startswith("_"):
            start = " ".join(words[:8])
            q = f'Which function in {path} has a docstring that begins with "{start}"?'
            out.append(_item("doc", q, f.name, f.start, f.qualname, extra={"prefix": start}))
    return out


def q_returns(path, tree, text) -> list:
    out = []
    for f in ctxlib.functions_of(path, text):
        r = _node_at(tree, f).returns
        ans = ast.unparse(r) if r is not None else ""
        if trivial_ok(ans) and len(ans) > 4:
            q = f'What is the return type annotation of the function "{f.qualname}" in {path}?'
            out.append(_item("returns", q, ans, f.start, f.qualname))
    return out


def q_imports(path, tree, text) -> list:
    out = []
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module != "__future__":
            mod = "." * n.level + (n.module or "")
            for a in n.names:
                q = f'In {path}, from which module is "{a.asname or a.name}" imported?'
                out.append(_item("import", q, mod, n.lineno, a.name))
    return out


def q_owners(files: list) -> list:
    """The class that defines a method whose name is defined nowhere else in the context."""
    defs = {}
    for path, text in files:
        for f in ctxlib.functions_of(path, text):
            defs.setdefault(f.name, []).append(f)
    out = []
    for name, fs in defs.items():
        f = fs[0]
        if len(fs) == 1 and "." in f.qualname and len(name) >= 6 and not name.startswith("_"):
            cls = f.qualname.rsplit(".", 1)[0].split(".")[-1]
            q = f'Which class defines the method "{name}"?'
            out.append({**_item("owner", q, cls, f.start, f.qualname), "src_path": f.path})
    return out


def q_callers(files: list) -> list:
    """For functions defined once in the context and called from exactly one function: that caller."""
    defs, calls = {}, {}
    for path, text in files:
        tree = ast.parse(text)
        for f in ctxlib.functions_of(path, text):
            defs.setdefault(f.name, []).append(f)
        for f in ctxlib.functions_of(path, text):
            for n in ast.walk(_node_at(tree, f)):
                if isinstance(n, ast.Call):
                    name = n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None)
                    if name:
                        calls.setdefault(name, []).append((path, f, n.lineno))
    out = []
    for name, ds in defs.items():
        sites = calls.get(name, [])
        callers = {(p, c.qualname) for p, c, _ in sites}
        if len(ds) == 1 and len(callers) == 1 and len(sites) <= 2 and len(name) >= 5 and not name.startswith("_"):
            path, caller, line = sites[0]
            q = f'Which function in the files above contains a call to "{name}"?'
            out.append({**_item("caller", q, caller.name, line, caller.qualname), "src_path": path})
    return out


def _item(kind, question, answer, line, subject, **extra) -> dict:
    base = {"kind": kind, "question": question, "answer": answer, "src_line": line, "subject": subject}
    return {"trivial": False, **base, **extra}


def all_questions(files: list) -> list:
    out = []
    for path, text in files:
        tree = ast.parse(text)
        for fn in (q_defaults, q_constants, q_bases, q_docs, q_returns, q_imports):
            out += [{**q, "src_path": path} for q in fn(path, tree, text)]
    out += q_callers(files) + q_owners(files)
    return unique_questions(out)


def unique_questions(qs: list) -> list:
    """Drop questions whose text occurs twice (same name in two classes) and docstring prefixes that repeat."""
    seen = {}
    for q in qs:
        seen.setdefault(q["question"], []).append(q)
    return [v[0] for v in seen.values() if len(v) == 1]


def add_positions(qs: list, ctx: dict) -> None:
    text = Path(common.WORK / "ctx" / f"{ctx['repo']}.txt").read_text()
    files = dict(ctx["files"])

    def pos(q):
        lines = files[q["src_path"]].splitlines(keepends=True)[: q["src_line"]]
        off = ctx["offsets"][q["src_path"]] + len("".join(lines))
        q["src_pos"] = common.count_tokens(text[:off])

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(pos, qs))


def pick_kind(pool: list, quota: int, rng: random.Random) -> list:
    """Quota questions of one kind; trivial answers (None, True, False) make up at most a fifth."""
    hard = [q for q in pool if not q["trivial"]]
    easy = [q for q in pool if q["trivial"]]
    pool = hard + rng.sample(easy, min(len(easy), max(1, quota // 5)))
    return rng.sample(pool, min(quota, len(pool)))


def pick(qs: list, ctx: dict, count: int, local_share: float) -> list:
    rng = random.Random(SEED)
    total = ctx["tokens"]
    far = [q for q in qs if q["src_pos"] <= total - MARGIN]
    near = [q for q in qs if q["src_pos"] > total - LOCAL_SPAN]
    n_local = min(len(near), round(count * local_share))
    chosen_local = rng.sample(near, n_local)
    by_kind = {}
    for q in far:
        by_kind.setdefault(q["kind"], []).append(q)
    quota = (count - n_local) // len(by_kind)
    chosen = []
    for k in sorted(by_kind):
        chosen += pick_kind(by_kind[k], quota, rng)
    rest = [q for q in far if q not in chosen]  # kinds with few items leave room; fill from the others
    chosen += rng.sample(rest, min(len(rest), count - n_local - len(chosen)))
    for q in chosen_local:
        q["local"] = True
    for q in chosen:
        q["local"] = False
    return chosen + chosen_local


def main():
    repo = sys.argv[1]
    count = int(sys.argv[2])
    share = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    ctx = common.read_json(common.WORK / "ctx" / f"{repo}-ctx.json")
    files = [(p, t) for p, t in ctx["files"]]
    qs = all_questions(files)
    add_positions(qs, ctx)
    chosen = pick(qs, ctx, count, share)
    out = common.WORK / "tasks" / f"{repo}-A.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    kinds = {}
    with out.open("w") as f:
        for i, q in enumerate(sorted(chosen, key=lambda q: q["src_pos"])):
            q.update(id=f"{repo}-A{i:03d}", family="A", repo=repo)
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
            key = q["kind"] + (" (local)" if q["local"] else "")
            kinds[key] = kinds.get(key, 0) + 1
    print(f"{repo}: {len(qs)} questions available, {len(chosen)} chosen {kinds}")


if __name__ == "__main__":
    main()
