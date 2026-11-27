"""The suites of bench/agent: each runs the model through a harness that ends in pass or fail and returns one record per
task: {"task": id, "ok": bool, ...}. A suite talks to a server that the job has started (url), never starts one itself.

Harnesses (all existing): SWE-bench Verified subset with mini-swe-agent and the swebench grader in rootless
Podman, the Aider-polyglot-style exercises, the 23 own Python tasks with tests, the 32 tool-call JSON prompts.
Seeds: the same base seed and repeat number give both arms the same random stream per task.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV = Path(os.environ.get("AGENT_VENV", "/var/tmp/claude/139/venv"))
POLYGLOT_DATA = Path(os.environ.get("AGENT_POLYGLOT", "/var/tmp/claude/139/polyglot-benchmark-main"))
CSHARP_DATA = Path(os.environ.get("AGENT_CSHARP", "/var/tmp/claude/139/polyglot-csharp"))  # exercism/csharp, MIT
CSHARP_VALID = Path(os.environ.get("AGENT_CSHARP_VALID", "/var/tmp/claude/139/valid-csharp.txt"))
POLYGLOT_VALID = Path(os.environ.get("AGENT_POLYGLOT_VALID", "/var/tmp/claude/139/valid.txt"))
# java and cpp from the Aider copy, php, typescript and dart from the Exercism tracks (bench/polyglot/prepare.py);
# one valid list per language (the reference check of research/measurements/2026-11-19-polyglot-languages.md)
LANGS_DATA = Path(os.environ.get("AGENT_LANGS_DATA", "/var/tmp/claude/poly-langs/data"))
LANGS_VALID = Path(os.environ.get("AGENT_LANGS_VALID", "/var/tmp/claude/poly-langs"))
NEW_LANGS = ("java", "cpp", "php", "typescript", "dart", "bash")
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}  # the model card's, as in every agent run since then

# a short agent loop: write, run the tests, read the output, fix
ATTEMPTS = int(os.environ.get("AGENT_ATTEMPTS", "4"))

OFFSET: dict = {}  # suite -> number of leading tasks skipped (a second night takes the next slice)

# suite -> server context (tokens)
CONTEXT = {"swe": 262144, "polyglot": 65536, "csharp": 65536, "effort": 16384, "json": 8192}
CONTEXT.update({lang: 65536 for lang in NEW_LANGS})


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def swe_ids(limit: int) -> list[str]:
    return (ROOT / "bench/swe/subset.txt").read_text().split()[:limit]


SWE_TASK_CAP_S = int(
    os.environ.get("AGENT_SWE_TASK_CAP_S", str(40 * 60))
)  # 5 x the 8.5 min median of the 2026-11-16 run
SWE_STOP_MARKERS = ("APITimeoutError", "Request timed out")  # a retry with the same seed repeats the same 600 s loop


def _swe_command(url: str, out: Path, seed: int, step_limit: int, task: str) -> list[str]:
    cmd = [str(VENV / "bin/mini-extra"), "swebench", "--subset", "verified", "--split", "test"]
    cmd += ["--filter", f"^({task})$", "-o", str(out / "swe"), "-m", "openai/default", "-c", "swebench.yaml"]
    for kv in (
        f"model.model_kwargs.api_base={url}/v1",
        "model.model_kwargs.api_key=none",
        f"model.model_kwargs.temperature={SAMPLING['temperature']}",
        f"model.model_kwargs.top_p={SAMPLING['top_p']}",
        f"model.model_kwargs.extra_body.top_k={SAMPLING['top_k']}",
        f"model.model_kwargs.extra_body.seed={seed}",
        "environment.executable=podman",
        "environment.container_timeout=5h",
        f"agent.step_limit={step_limit}",
        "environment.timeout=120",
    ):
        cmd += ["-c", kv]
    return cmd


def _swe_task(cmd: list[str], env: dict) -> str:
    """Runs one task; returns "" when it ended on its own, else why it was stopped. The first client timeout stops the
    task (the retry is identical, a repetition loop), as does the wall-clock cap; the container is removed."""
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
    )
    names: set[str] = set()
    why = ""
    t0 = time.time()
    for line in proc.stdout:
        names.update(re.findall(r"minisweagent-[0-9a-f]{8}", line))
        if any(m in line for m in SWE_STOP_MARKERS):
            why = "loop/timeout: client timeout"
        elif time.time() - t0 > SWE_TASK_CAP_S:
            why = "loop/timeout: wall-clock cap"
        if why:
            break
    if why:
        os.killpg(proc.pid, signal.SIGTERM)
        for n in names:
            subprocess.run(["podman", "rm", "-f", n], capture_output=True, check=False)
    proc.wait()
    return why


def swe(url: str, out: Path, seed: int, limit: int, step_limit: int = 100) -> list[dict]:
    """SWE-bench Verified subset (bench/swe/subset.txt): mini-swe-agent plays each task in its own call (tasks that
    already have a prediction are skipped, so an interrupted arm resumes), then the harness grades the patches.
    A stopped task (loop/timeout) is a model failure, listed in swe.loops.jsonl and counted apart from infrastructure
    failures. Grading needs no server and is run by the caller after the server stopped (see grade_swe)."""
    ids = swe_ids(limit)
    cfg = out / "msa-config"
    cfg.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MSWEA_GLOBAL_CONFIG_DIR": str(cfg), "MSWEA_COST_TRACKING": "ignore_errors"}
    loops = out / "swe.loops.jsonl"
    done = {r["task"] for r in _lines(loops)}
    for task in ids:
        if task in done:
            continue
        why = _swe_task(_swe_command(url, out, seed, step_limit, task), env)
        if why:
            with loops.open("a") as f:
                f.write(json.dumps({"task": task, "reason": why, "at": time.strftime("%H:%M:%S")}) + "\n")
    return [{"task": i, "ok": None} for i in ids]  # graded later


def grade_swe(out: Path, ids: list[str]) -> list[dict]:
    preds = out / "swe/preds.json"
    ev = out / "swe.eval.jsonl"
    if preds.exists():
        subprocess.run(
            [str(VENV / "bin/python"), "-I", str(ROOT / "bench/swe/evaluate.py"), str(preds), str(ev)], check=False
        )
    got = {r["id"]: bool(r.get("resolved")) for r in _lines(ev) if "id" in r}
    loops = {r["task"]: r["reason"] for r in _lines(out / "swe.loops.jsonl")}
    return [{"task": i, "ok": got.get(i, False), **({"reason": loops[i]} if i in loops else {})} for i in ids]


def polyglot(url: str, out: Path, seed: int, limit: int, *, langs: str = "") -> list[dict]:
    """Aider-polyglot-style exercises, answer with whole files, test output fed back (up to ATTEMPTS tries
    in one conversation); ok = passed within the attempts, ok1 = passed on the first.
    Fixed shuffled order, so both arms see the same prefix."""
    res = out / "polyglot.jsonl"
    data, valid = (CSHARP_DATA, CSHARP_VALID) if langs == "csharp" else (POLYGLOT_DATA, POLYGLOT_VALID)
    if langs in NEW_LANGS:
        data, valid = LANGS_DATA, LANGS_VALID / f"valid-{langs}.txt"
    cmd = [sys.executable, "-I", str(ROOT / "bench/polyglot/run.py"), "run", "--data", str(data)]
    cmd += ["--valid", str(valid), "--url", url, "--limit", str(limit), "--out", str(res)]
    cmd += ["--start", str(OFFSET.get("polyglot", 0) if not langs else 0)]
    cmd += ["--langs", langs] if langs else []
    cmd += ["--sampling", json.dumps({**SAMPLING, "seed": seed}), "--attempts", str(ATTEMPTS)]
    subprocess.run(cmd, check=False)
    return [{"task": r["id"], "ok": bool(r["passed2"]), "ok1": bool(r["passed1"])} for r in _lines(res)]


def csharp(url: str, out: Path, seed: int, limit: int) -> list[dict]:
    """The C# (.NET) practice exercises of Exercism, the same method as polyglot (own suite: the polyglot order and
    its 72-exercise prefix of night 1 must stay as they are)."""
    return polyglot(url, out, seed, limit, langs="csharp")


def _new_lang(lang: str):
    def runner(url: str, out: Path, seed: int, limit: int) -> list[dict]:
        return polyglot(url, out, seed, limit, langs=lang)

    runner.__doc__ = f"The {lang} exercises, the same method as polyglot, own suite."
    return runner


def effort(url: str, out: Path, seed: int, limit: int) -> list[dict]:
    """The 23 own Python tasks with tests (scripts/bench-effort.py), level high, sampling as above."""
    env = {**os.environ, "SAMPLING": f"{SAMPLING['temperature']},{SAMPLING['top_p']},{SAMPLING['top_k']},{seed}"}
    env["MAX_TOKENS"] = "24000"
    log = out / "effort.log"
    with log.open("w") as f:
        subprocess.run(
            [sys.executable, "-I", str(ROOT / "scripts/bench-effort.py"), url, "high"],
            env=env,
            cwd=out,
            stdout=f,
            check=False,
        )
    files = sorted((out / "work/effort").glob("*.jsonl"))
    return [{"task": r["task"], "ok": bool(r["ok"])} for r in _lines(files[-1])][:limit] if files else []


def tool_json(url: str, out: Path, seed: int, limit: int) -> list[dict]:
    """The 32 tool-call JSON prompts (greedy, the seed does not matter): valid = a parseable tool call."""
    script = ROOT / "bench/agent/toolcall32.py"
    p = subprocess.run([sys.executable, "-I", str(script), url], capture_output=True, text=True, check=False)
    rows = [x for x in p.stdout.splitlines() if x[:2] in ("0 ", "1 ")]
    return [{"task": f"json{i:02d}", "ok": x[0] == "1"} for i, x in enumerate(rows[:limit])]


RUNNERS = {"swe": swe, "polyglot": polyglot, "csharp": csharp, "effort": effort, "json": tool_json}
RUNNERS.update({lang: _new_lang(lang) for lang in NEW_LANGS})
