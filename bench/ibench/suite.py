"""Measurement suites: TOML files with cases. A case becomes a sequence of request bodies per run."""

import copy
import json
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent
CORPUS = BENCH / "corpus" / "python-stdlib-3.14.txt"
SYSTEM = "You are a precise coding assistant."


@dataclass
class Case:
    name: str
    description: str = ""
    runs: int = 5
    warmup: int = 1
    # cold: nonce at the start, so no prefix hit is possible.
    # warm: same prefix in every run, only the end of the question changes; the warm-up run fills the cache.
    cache: str = "cold"
    temperature: float = 0.0
    max_tokens: int = 512
    expect: str | None = None  # regex on the answer text
    min_decode_tokens: int = 64  # below this the client decode rate is not meaningful
    pause_s: float = 0.0  # pause after each run
    extra: dict = field(default_factory=dict)  # additional body fields
    # exactly one source: prompt, context_chars + question, or trace
    prompt: str | None = None
    context_chars: int = 0
    needle: str | None = None
    needle_depth: float = 0.5
    question: str | None = None
    trace: str | None = None


def load(path) -> tuple[dict, list[Case]]:
    path = Path(path)
    data = tomllib.loads(path.read_text())
    defaults = data.get("defaults", {})
    known = {f.name for f in fields(Case)}
    cases = []
    for raw in data.get("case", []):
        merged = {**defaults, **raw}
        merged["extra"] = {**defaults.get("extra", {}), **raw.get("extra", {})}
        unknown = set(merged) - known
        if unknown:
            raise ValueError(f"{path}: case {raw.get('name')!r}: unknown fields {sorted(unknown)}")
        c = Case(**merged)
        sources = [c.prompt is not None, bool(c.context_chars), c.trace is not None]
        if sum(sources) != 1:
            raise ValueError(f"{path}: case {c.name!r}: exactly one of prompt, context_chars, trace")
        if c.context_chars and not c.question:
            raise ValueError(f"{path}: case {c.name!r}: context_chars needs question")
        if c.cache not in ("cold", "warm"):
            raise ValueError(f"{path}: case {c.name!r}: cache must be cold or warm")
        if c.cache == "warm" and c.warmup < 1:
            raise ValueError(f"{path}: case {c.name!r}: warm needs warmup >= 1")
        if c.trace:
            c.trace = str((path.parent / c.trace).resolve())
        cases.append(c)
    names = [c.name for c in cases]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: duplicate case names")
    meta = {k: v for k, v in data.items() if k not in ("defaults", "case")}
    meta.setdefault("name", path.stem)
    return meta, cases


_corpus_cache: str | None = None


def context(chars: int, needle: str | None = None, depth: float = 0.5) -> str:
    """The first `chars` characters of the corpus, cut at a line boundary, with an optional needle."""
    global _corpus_cache
    if _corpus_cache is None:
        _corpus_cache = CORPUS.read_text()
    text = _corpus_cache
    if chars > len(text):
        raise ValueError(f"Corpus has only {len(text)} characters, {chars} requested")
    cut = text.rfind("\n", 0, chars) + 1 or chars
    body = text[:cut]
    if needle:
        pos = body.rfind("\n", 0, int(len(body) * depth)) + 1
        body = body[:pos] + f'SECRET_DEPLOY_TOKEN = "{needle}"  # used by the release script\n' + body[pos:]
    return body


def _user_text(case: Case, run: int) -> str:
    if case.prompt is not None:
        text = case.prompt
    else:
        ctx = context(case.context_chars, case.needle, case.needle_depth)
        text = f"Here is a source file:\n\n```python\n{ctx}```\n\n{case.question}"
    if case.cache == "warm":
        text += f"\n\n(request {run})"
    return text


def _prefix_nonce_openai(body: dict, nonce: str) -> None:
    msgs = body.setdefault("messages", [])
    tag = f"[run {nonce}]\n"
    if msgs and msgs[0].get("role") in ("system", "developer"):
        c = msgs[0].get("content")
        if isinstance(c, list):
            msgs[0]["content"] = [{"type": "text", "text": tag}, *c]
        else:
            msgs[0]["content"] = tag + (c or "")
    else:
        msgs.insert(0, {"role": "system", "content": tag})


def _prefix_nonce_anthropic(body: dict, nonce: str) -> None:
    tag = f"[run {nonce}]\n"
    s = body.get("system")
    if isinstance(s, list):
        body["system"] = [{"type": "text", "text": tag}, *s]
    else:
        body["system"] = tag + (s or "")


def load_trace(path: str) -> list[dict]:
    """Recorded requests (bench/ibench/proxy.py), one JSON line per request."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    if not out:
        raise ValueError(f"{path}: empty trace")
    return out


def bodies(case: Case, api: str, model: str, run: int, nonce: str | None, target_extra: dict) -> list[dict]:
    """The request bodies of one run: a trace yields several, anything else one."""
    if case.trace:
        steps = []
        for rec in load_trace(case.trace):
            if rec.get("api") != api:
                raise ValueError(f"Trace {case.trace}: recorded with API {rec.get('api')!r}, target speaks {api!r}")
            b = copy.deepcopy(rec["body"])
            b["model"] = model
            b.update(target_extra)
            b.update(case.extra)
            if nonce:
                (_prefix_nonce_openai if api == "openai" else _prefix_nonce_anthropic)(b, nonce)
            steps.append(b)
        return steps

    system = (f"[run {nonce}]\n" if nonce else "") + SYSTEM
    user = _user_text(case, run)
    b = {"model": model, "max_tokens": case.max_tokens, "temperature": case.temperature}
    if api == "openai":
        b["messages"] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    else:
        b["system"] = system
        b["messages"] = [{"role": "user", "content": user}]
    b.update(target_extra)
    b.update(case.extra)
    return [b]
