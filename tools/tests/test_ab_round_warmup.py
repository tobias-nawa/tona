"""scripts/ab.py: the discarded start per arm and round (macOS default) and decisions that the interval of all rounds
contradicts."""

import argparse

import ab
import ab_report
from ab_stats import FASTER, Plan

# decode-text of 2026-11-12-mac-iq3xxs-repack.md: B against A per round, percent
TEXT = [1.73, 7.20, -5.67, 6.78, 1.69, -1.26]


def rounds_of(diffs: list) -> list:
    return [{"A": {"decode-text": 100.0}, "B": {"decode-text": 100.0 + d}, "order": "AB"} for d in diffs]


def test_an_early_decision_that_all_rounds_contradict_is_open():
    k, look = ab_report.first_decision(ab_report.diffs_of(rounds_of(TEXT), "decode-text"), Plan())
    assert k == 2 and look.decision == FASTER  # the sequential rule alone decides at round 2
    k, look, final = ab_report.decide(rounds_of(TEXT), Plan())["decode-text"]
    assert k is None and final.ci_diff[0] < 0 < final.ci_diff[1]
    assert ab_report.decision_text(k, look).startswith("open")


def test_an_early_decision_that_all_rounds_confirm_stands():
    k, look, _ = ab_report.decide(rounds_of([5.0, 5.2, 4.9, 5.1]), Plan())["decode-text"]
    assert k == 2 and look.decision == FASTER


def test_round_warmup_measures_each_arm_twice_and_keeps_the_second():
    calls = []

    def measure(key):
        calls.append(key)
        seen = calls.count(key)
        return {"decode@1": 100.0 if seen % 2 == 0 else 1.0}, {"1": "s"}  # discarded starts give 1.0

    rounds = []
    ab.run_rounds(measure, Plan(max_rounds=2), rounds, stop=ab.Stop(round_warmup=True, fixed=True))
    assert calls == ["A", "A", "B", "B", "B", "B", "A", "A"]
    assert all(r[k]["decode@1"] == 100.0 for r in rounds for k in "AB")


def test_round_warmup_defaults_to_macos_only(monkeypatch):
    args = argparse.Namespace(round_warmup="auto")
    for platform, want in (("darwin", True), ("linux", False)):
        monkeypatch.setattr(ab.sys, "platform", platform)
        assert ab.round_warmup(args) is want
    monkeypatch.setattr(ab.sys, "platform", "linux")
    assert ab.round_warmup(argparse.Namespace(round_warmup="on"))
    monkeypatch.setattr(ab.sys, "platform", "darwin")
    assert not ab.round_warmup(argparse.Namespace(round_warmup="off"))


def test_a_round_with_a_backup_is_repeated_and_kept_aside():
    backups = iter(["a system backup ran", None, None])

    def measure(key):
        return {"decode@1": 100.0}, {"1": "s"}

    rounds, gone = [], []
    stop = ab.Stop(fixed=True, discard=lambda: next(backups), discarded=gone)
    ab.run_rounds(measure, Plan(max_rounds=2), rounds, stop=stop)
    assert len(rounds) == 2 and [r["order"] for r in rounds] == ["AB", "BA"]  # round 1 again, then round 2
    assert len(gone) == 1 and gone[0]["discarded"] == "a system backup ran"
    data = {
        "config": {"A": {"build": "a", "env": {}}, "B": {"build": "b", "env": {}}, "deterministic": False},
        "rounds": rounds,
        "discarded": gone,
        "stop_reason": "x",
        "wall_seconds": 1.0,
    }
    assert "1 discarded" in ab_report.markdown(data, Plan(max_rounds=2))
