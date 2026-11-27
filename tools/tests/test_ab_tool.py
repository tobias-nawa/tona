"""scripts/ab.py without GPU: parsers, rounds with a fake measurement, report, lock, process handling, corpus."""

import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import ClassVar

import ab
import ab_report
import ab_server
import make_corpus
import pytest
from ab_server import Arm, ServerConfig
from ab_stats import Plan
from ab_workloads import Request
from conftest import commit_all, write

BENCH_SET_OUTPUT = (
    "Code: 4400 Tokens in 50.3 s = 87.5 tok/s decode (per prompt 70-110), 2.61 tokens per pass, "
    "1500/2400 drafts accepted\n"
    "Text: 1200 Tokens in 18.8 s = 63.9 tok/s decode (per prompt 55-70)\n"
)


def test_parse_bench_set_reads_both_groups():
    assert ab_server.parse_bench_set(BENCH_SET_OUTPUT) == {"decode-code": 87.5, "decode-text": 63.9}


def test_parse_bench_set_rejects_other_output():
    with pytest.raises(ValueError):
        ab_server.parse_bench_set("Code: 12 Tokens in 1.0 s")


def test_parse_completion_names_prefill_and_decode():
    resp = {
        "timings": {"prompt_n": 30000, "prompt_ms": 20000.0, "predicted_per_second": 71.25},
        "choices": [{"message": {"content": "hello"}}],
    }
    got = ab_server.parse_completion(resp)
    assert got["prefill"] == pytest.approx(1500.0)
    assert got["decode"] == 71.25
    assert got["sha"] == "2cf24dba5fb0a30e"  # sha256("hello")[:16]


def test_run_context_asks_once_per_length_and_caches_prompts(monkeypatch):
    calls = []

    def fake_post(url, path, body, timeout=0):
        calls.append(path)
        if path.endswith("count_tokens"):
            return {"input_tokens": len(body["messages"][0]["content"]) // 4}
        return {
            "timings": {"prompt_n": 100, "prompt_ms": 100.0, "predicted_per_second": 50.0},
            "choices": [{"message": {"content": "x"}}],
        }

    monkeypatch.setattr(ab_server, "post", fake_post)
    prompts = {}
    corpora = ab_server.Corpora("a" * 40000)
    metrics, shas = ab_server.run_context("u", corpora, [Request("sum", 2000, 200), Request("sum", 5000, 200)], prompts)
    assert set(metrics) == {"prefill@2000", "decode@2000", "prefill@5000", "decode@5000"}
    assert set(shas) == {"2000", "5000"}
    before = len(calls)
    ab_server.run_context("u", corpora, [Request("sum", 2000, 200)], prompts)
    assert len(calls) == before + 1  # only the completion, the bisection is cached


def test_prose_requests_are_cut_from_the_prose_corpus(monkeypatch):
    monkeypatch.setattr(ab_server, "count_tokens", lambda url, text: len(text) // 4)
    monkeypatch.setattr(
        ab_server,
        "post",
        lambda *a, **k: {
            "timings": {"prompt_n": 1, "prompt_ms": 1.0, "predicted_per_second": 1.0},
            "choices": [{"message": {"content": "x"}}],
        },
    )
    corpora = ab_server.Corpora("c" * 400000, "p" * 400000)
    prompts = {}
    metrics, _ = ab_server.run_context("u", corpora, [Request("prose", 2000, 400)], prompts)
    assert set(metrics) == {"prefill@2000:prose", "decode@2000:prose"}
    assert "p" in prompts[("prose", 2000)] and "c" * 10 not in prompts[("prose", 2000)]


def test_prompt_for_stays_below_target(monkeypatch):
    def count(text):
        return len(text) // 4

    monkeypatch.setattr(ab_server, "count_tokens", lambda url, text: count(text))
    prompt = ab_server.prompt_for("u", "b" * 100_000, 10_000)
    assert count(prompt) < 10_000 <= count(prompt) + 2000 // 4 + 1


def test_forbidden_port_is_refused():
    with pytest.raises(SystemExit):
        ab_server.check_port(8662)


def test_child_env_drops_inherited_tona_variables_and_applies_arms(monkeypatch, tmp_path):
    monkeypatch.setenv("TONA_DIAG", "1")
    monkeypatch.setenv("KEEP", "yes")
    cfg = ServerConfig(tmp_path, "m", "d", 1000, base_env={"TONA_DETERMINISTIC": "1", "OMP_NUM_THREADS": "5"})
    env = ab_server.child_env(cfg, Arm("B", "x", {"OMP_NUM_THREADS": "3"}))
    assert "TONA_DIAG" not in env and env["KEEP"] == "yes"
    assert env["TONA_DETERMINISTIC"] == "1" and env["OMP_NUM_THREADS"] == "3"


def test_child_env_picks_the_profile_by_model_id_and_an_arm_can_drop_it(tmp_path):
    (tmp_path / "toolchain/profiles").mkdir(parents=True)
    (tmp_path / "toolchain/profiles/m-one.bin").write_bytes(b"x")
    cfg = ServerConfig(tmp_path, str(tmp_path / "work/models/m-one/f.gguf"), "", 1000)
    profile = str(tmp_path / "toolchain/profiles/m-one.bin")
    assert ab_server.child_env(cfg, Arm("A", "x"))["TONA_EXPERT_PROFILE"] == profile
    assert "TONA_EXPERT_PROFILE" not in ab_server.child_env(cfg, Arm("A", "x", {"TONA_EXPERT_PROFILE": "none"}))
    # an id that only contains the name of a profile gets none; neither does a config without profiles
    other = Arm("B", "x", model=str(tmp_path / "work/models/m-one-q6/f.gguf"))
    assert "TONA_EXPERT_PROFILE" not in ab_server.child_env(cfg, other)
    off = ServerConfig(tmp_path, cfg.model, "", 1000, expert_profile=False)
    assert "TONA_EXPERT_PROFILE" not in ab_server.child_env(off, Arm("A", "x"))


def test_server_command_has_scope_memory_limit_no_swap_and_port(tmp_path, monkeypatch):
    monkeypatch.setattr(ab_server.sys, "platform", "linux")
    cfg = ServerConfig(tmp_path, "m.gguf", "d.gguf", 140000, port=8123, memory_max="54G")
    cmd = ab_server.server_command(cfg, Arm("A", "merged"))
    assert cmd[:4] == ["systemd-run", "--user", "--scope", "-q"]
    assert "MemoryMax=54G" in cmd and "MemorySwapMax=0" in cmd
    assert cmd[cmd.index("--port") + 1] == "8123"
    assert str(tmp_path / "work/build/merged/tona-server") in cmd


def test_build_dir_accepts_names_and_paths(tmp_path):
    assert ab_server.build_dir(tmp_path, Arm("A", "w1")) == tmp_path / "work/build/w1"
    assert ab_server.build_dir(tmp_path, Arm("A", "/opt/b")) == Path("/opt/b")


def test_stop_server_ends_the_process_by_pid():
    proc = subprocess.Popen(["sleep", "60"])
    ab_server.stop_server(proc)
    assert proc.poll() is not None
    ab_server.stop_server(proc)  # a finished process is fine
    ab_server.stop_server(None)


def test_stop_server_kills_after_the_grace_period():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"]
    )
    time.sleep(0.5)
    ab_server.stop_server(proc, grace=0.5)
    assert proc.poll() is not None


def test_wait_ready_reports_an_exited_server():
    proc = subprocess.Popen(["true"])
    proc.wait()
    with pytest.raises(RuntimeError, match="exited"):
        ab_server.wait_ready(proc, 1, 5)


def test_parse_env():
    assert ab.parse_env(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}
    with pytest.raises(SystemExit):
        ab.parse_env(["nope"])


def fake_measure(effect: float, shas=("aa", "aa")):
    """A measurement with fixed values: B differs from A by `effect` percent. shas: text hash of A and B."""

    def measure(key):
        value = 100.0 * (1 + effect / 100 if key == "B" else 1)
        return {"decode@30000": value, "prefill@30000": 1000.0}, {"30000": shas[key == "B"]}

    return measure


def test_order_alternates_while_a_metric_stays_open():
    rounds = []
    ab.run_rounds(fake_measure(0.0), Plan(max_rounds=4, prior_sd=3.0), rounds)
    assert [r["order"] for r in rounds] == ["AB", "BA", "AB", "BA"]


def test_all_metrics_clear_stops_at_two():
    def measure(key):
        f = 0.9 if key == "B" else 1.0
        return {"decode@30000": 100 * f, "prefill@30000": 1000 * f}, {"30000": "s"}

    rounds = []
    reason = ab.run_rounds(measure, Plan(), rounds)
    assert len(rounds) == 2 and reason == "all metrics decided after round 2"


def test_rounds_survive_an_error_in_a_later_round():
    calls = {"n": 0}
    inner = fake_measure(0.0)

    def measure(key):
        calls["n"] += 1
        if calls["n"] == 5:
            raise RuntimeError("server died")
        return inner(key)

    rounds = []
    with pytest.raises(RuntimeError):
        ab.run_rounds(measure, Plan(), rounds)
    assert len(rounds) == 2


def data_for(rounds, deterministic=False):
    env = {"TONA_DETERMINISTIC": "1"} if deterministic else {}
    config = {"A": {"build": "merged", "env": env}, "B": {"build": "w", "env": env}, "deterministic": deterministic}
    return {"config": config, "rounds": rounds, "stop_reason": "test", "wall_seconds": 600.0}


def test_report_lists_every_round_and_names_rates():
    rounds = []
    ab.run_rounds(fake_measure(-10.0), Plan(max_rounds=3), rounds)
    text = ab_report.markdown(data_for(rounds), Plan(max_rounds=3))
    assert "Decode tok/s" in text and "Prefill tok/s" in text
    assert text.count("| AB |") + text.count("| BA |") == 3
    assert "**B slower** after round 2" in text
    assert "fixed 3 rounds" in text


def test_report_warns_about_diverging_hashes_in_deterministic_mode():
    rounds = []
    ab.run_rounds(fake_measure(-10.0, shas=("aa", "bb")), Plan(max_rounds=2), rounds)
    assert "WARNING" in ab_report.markdown(data_for(rounds, deterministic=True), Plan(max_rounds=2))
    assert "WARNING" not in ab_report.markdown(data_for(rounds, deterministic=False), Plan(max_rounds=2))


def test_report_confirms_identical_hashes():
    rounds = []
    ab.run_rounds(fake_measure(-10.0), Plan(max_rounds=2), rounds)
    text = ab_report.markdown(data_for(rounds, deterministic=True), Plan(max_rounds=2))
    assert "WARNING" not in text and "| yes | yes | yes |" in text


def test_json_has_rounds_and_decisions():
    rounds = []
    ab.run_rounds(fake_measure(-10.0), Plan(max_rounds=2), rounds)
    out = json.loads(ab_report.to_json(data_for(rounds), Plan(max_rounds=2)))
    assert len(out["rounds"]) == 2
    assert out["decisions"]["decode@30000"]["decision"] == "B slower"
    assert out["decisions"]["prefill@30000"]["decision"] in ("open", "equal")


def test_empty_report():
    assert ab_report.markdown(data_for([]), Plan()) == "No round completed.\n"


def test_gpu_lock_excludes_a_second_holder(tmp_path):
    path = str(tmp_path / "gpu.lock")
    with ab.GpuLock(path, True), open(path, "a") as other, pytest.raises(BlockingIOError):
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with open(path, "a") as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_gpu_lock_can_be_disabled(tmp_path):
    with ab.GpuLock(str(tmp_path / "none"), False):
        assert not (tmp_path / "none").exists()


def test_gpu_lock_tells_the_children_that_it_is_held(tmp_path, monkeypatch):
    monkeypatch.delenv("TONA_GPU_LOCK_HELD", raising=False)
    with ab.GpuLock(str(tmp_path / "gpu.lock"), True):
        assert os.environ["TONA_GPU_LOCK_HELD"] == "1"
    assert "TONA_GPU_LOCK_HELD" not in os.environ


def test_a_held_lock_in_the_environment_acts_like_no_lock(monkeypatch):
    args = ab.parse_args(["--a", "x", "--b", "y"])
    monkeypatch.delenv("TONA_GPU_LOCK_HELD", raising=False)
    assert ab.lock_wanted(args)
    monkeypatch.setenv("TONA_GPU_LOCK_HELD", "1")
    assert not ab.lock_wanted(args)
    monkeypatch.delenv("TONA_GPU_LOCK_HELD")
    assert not ab.lock_wanted(ab.parse_args(["--a", "x", "--b", "y", "--no-lock"]))


def test_plan_rejects_bad_round_limits():
    args = ab.parse_args(["--a", "x", "--b", "y", "--min-rounds", "5", "--max-rounds", "3"])
    with pytest.raises(SystemExit):
        ab.plan_from(args)


def test_corpus_is_built_from_the_commit_not_the_working_tree(repo):
    write(repo, "engine/a.cpp", "int a;\n")
    write(repo, "engine/third_party/x.cpp", "vendored\n")
    write(repo, "scripts/b.py", "x = 1\n")
    write(repo, "README.md", "doc\n")
    commit_all(repo)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    (repo / "engine/a.cpp").write_text("int changed;\n")
    text = make_corpus.build(sha)
    sources = (
        "// ===== inference-engine/engine/a.cpp =====\nint a;\n\n"
        "// ===== inference-engine/scripts/b.py =====\nx = 1\n\n"
    )
    # sources first, documents appended: the sources stay the exact prefix, so their prompts never move
    assert text == sources + "// ===== inference-engine/README.md =====\ndoc\n\n"


def test_port_in_use_is_refused_after_the_wait():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        with pytest.raises(SystemExit, match="in use"):
            ab_server.check_port(s.getsockname()[1], wait=0.2)


def test_primary_metrics_decide_when_to_stop():
    def measure(key):
        f = 0.9 if key == "B" else 1.0  # decode clearly slower, prefill identical (stays open)
        return {"decode@30000": 100 * f, "prefill@30000": 1000.0}, {"30000": "s"}

    everything, only_decode = [], []
    ab.run_rounds(measure, Plan(max_rounds=4), everything)
    reason = ab.run_rounds(measure, Plan(max_rounds=4), only_decode, stop=ab.Stop(primary=["decode"]))
    assert len(everything) == 4 and len(only_decode) == 2
    assert reason == "all metrics decided after round 2"


def test_warmup_is_not_a_round(monkeypatch, tmp_path):
    seen = []

    class Fake:
        loads: ClassVar[list] = []  # ab_load summaries per start (Measurer.loads)

        def __call__(self, key):
            seen.append(key)
            return {"decode@1": 1.0}, {}

        def close(self):
            pass

    monkeypatch.setattr(ab, "Measurer", lambda *a, **k: Fake())
    monkeypatch.setattr(ab, "load_corpus", lambda p: "x")
    monkeypatch.setattr(ab.ab_server, "check_port", lambda *a, **k: None)
    out = tmp_path / "o"
    argv = [
        "--workload",
        "server",
        "--a",
        "x",
        "--b",
        "y",
        "--round-warmup",
        "off",
        "--no-lock",
        "--targets",
        "1",
        "--max-rounds",
        "2",
        "--why-size",
        "t",
        "--out",
        str(out),
    ]
    argv += ["--expect", "decode:-1..1%", "--expect", "prefill:-1..1%", "--expect-source", "t"]
    ab.main(argv)
    assert seen[0] == "A" and seen[1:] == ["A", "B", "B", "A"]
    assert len(json.loads(out.with_suffix(".json").read_text())["rounds"]) == 2


def test_prompt_bisection_is_unchanged_by_appended_corpus_text(monkeypatch):
    """The prompt for a target that fits into the sources must not move when documents are appended."""
    monkeypatch.setattr(ab_server, "count_tokens", lambda url, text: len(text) // 4)
    base = "x" * 120_000
    plain = ab_server.prompt_for("u", base, 10_000, "sum", base_chars=len(base))
    extended = ab_server.prompt_for("u", base + "y" * 200_000, 10_000, "sum", base_chars=len(base))
    assert plain == extended
    beyond = ab_server.prompt_for("u", base + "y" * 200_000, 40_000, "sum", base_chars=len(base))
    assert "y" in beyond and len(beyond) // 4 < 40_000


def test_prompt_for_a_corpus_that_is_too_short_says_so(monkeypatch):
    monkeypatch.setattr(ab_server, "count_tokens", lambda url, text: len(text) // 4)
    with pytest.raises(RuntimeError, match="too short"):
        ab_server.prompt_for("u", "x" * 10_000, 50_000)


def test_write_request_reads_other_text_than_the_summary(monkeypatch):
    monkeypatch.setattr(ab_server, "count_tokens", lambda url, text: len(text) // 4)
    corpus = "a" * 1_000_000 + "Q" * 400_000
    s = ab_server.prompt_for("u", corpus, 5_000, "sum")
    w = ab_server.prompt_for("u", corpus, 5_000, "write")
    assert "Q" not in s and "Q" in w and s.split("\n")[0] != w.split("\n")[0]


def test_report_compares_with_the_reference_sha_of_main():
    rounds = []
    ab.run_rounds(fake_measure(-10.0, shas=("aa", "aa")), Plan(max_rounds=2), rounds)
    cfg = data_for(rounds, deterministic=True)
    cfg["config"]["reference_sha"] = {"30000": "aa"}
    assert "aa (B = ref)" in ab_report.markdown(cfg, Plan(max_rounds=2))
    cfg["config"]["reference_sha"] = {"30000": "zz"}
    text = ab_report.markdown(cfg, Plan(max_rounds=2))
    assert "B DIFFERS" in text and "WARNING" in text


def test_reference_sha_only_in_deterministic_mode_with_the_reference_corpus(monkeypatch):
    monkeypatch.setattr(make_corpus, "DEFAULT_SHA256", "abc")
    det = {"A": {"TONA_DETERMINISTIC": "1"}, "B": {"TONA_DETERMINISTIC": "1"}}
    assert ab.reference_sha("abc", det) != {} and "30000" in ab.reference_sha("abc", det)
    assert ab.reference_sha("other", det) == {}
    assert ab.reference_sha("abc", {"A": {}, "B": {}}) == {}


def test_expert_counts_are_the_difference_of_the_status_counters():
    before = {"requests": 3, "expert_hits": 100, "expert_misses": 10}
    after = {"requests": 4, "expert_hits": 180, "expert_misses": 12}
    assert ab_server.expert_counts(before, after) == {"expert_hits": 80, "expert_misses": 2}
    assert ab_server.expert_counts({}, {}) == {"expert_hits": 0, "expert_misses": 0}  # no totals yet


def test_pass_counts_and_rows_give_tokens_per_pass():
    before = {"passes": 10, "gen_tokens": 30, "mtp_passes": 20}
    after = {"passes": 110, "gen_tokens": 330, "mtp_passes": 220}
    c = ab_server.pass_counts(before, after)
    assert c == {"passes": 100, "gen_tokens": 300, "mtp_passes": 200}
    row = {"expert_hits": 1, "expert_misses": 0, **c}
    rounds = [{"counters": {"A": {"r": row}, "B": {"r": row}}}]
    assert ab_report.pass_rows(rounds)[-1] == "| r | 3.00 (2.00) | 3.00 (2.00) |"


def test_totals_after_waits_until_the_request_is_booked(monkeypatch):
    answers = iter([{"requests": 1}, {"requests": 1}, {"requests": 2, "expert_hits": 5}])
    monkeypatch.setattr(ab_server, "status_totals", lambda url: next(answers))
    assert ab_server.totals_after("u", 1, wait=5)["expert_hits"] == 5


def test_totals_after_gives_up_after_the_wait(monkeypatch):
    monkeypatch.setattr(ab_server, "status_totals", lambda url: {"requests": 1})
    assert ab_server.totals_after("u", 1, wait=0.3) == {"requests": 1}


def test_run_context_collects_expert_counters(monkeypatch):
    state = {"requests": 0, "hits": 0}

    def fake_post(url, path, body, timeout=0):
        if path.endswith("count_tokens"):
            return {"input_tokens": len(body["messages"][0]["content"]) // 4}
        state["requests"] += 1
        state["hits"] += 40
        return {
            "timings": {"prompt_n": 100, "prompt_ms": 100.0, "predicted_per_second": 50.0},
            "choices": [{"message": {"content": "x"}}],
        }

    monkeypatch.setattr(ab_server, "post", fake_post)
    monkeypatch.setattr(
        ab_server, "status_totals", lambda url: {"requests": state["requests"], "expert_hits": state["hits"]}
    )
    counters = {}
    ab_server.run_context("u", ab_server.Corpora("a" * 40000), [Request("sum", 2000, 200)], {}, None, counters)
    assert counters == {"2000": {"expert_hits": 40, "expert_misses": 0, "passes": 0, "gen_tokens": 0, "mtp_passes": 0}}


def test_report_shows_whether_the_expert_counters_are_identical():
    def counts(h):
        return {"2000": {"expert_hits": h, "expert_misses": 3}}

    rounds = [
        {"counters": {"A": counts(10), "B": counts(10)}},
        {"counters": {"A": counts(10), "B": counts(10)}},
    ]
    assert ab_report.counter_rows(rounds)[-1].endswith("| yes |")
    rounds[1]["counters"]["B"] = counts(11)
    assert ab_report.counter_rows(rounds)[-1].endswith("| NO |")
    assert ab_report.counter_rows([{}]) == []


def test_rounds_keep_the_counters_of_the_measurement():
    def measure(key):
        return {"decode@1": 1.0}, {}, {"1": {"expert_hits": 7, "expert_misses": 1}}

    rounds = []
    ab.run_rounds(measure, Plan(max_rounds=2), rounds)
    assert rounds[0]["counters"]["A"]["1"]["expert_hits"] == 7


def test_fixed_rounds_do_not_stop_early():
    rounds = []
    ab.run_rounds(fake_measure(-10.0), Plan(max_rounds=4), rounds, stop=ab.Stop(fixed=True))
    assert len(rounds) == 4
