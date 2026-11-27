"""Known noise: table, use in the stop rule (one round may decide), simulation, proposal of updates."""

import random

import ab
import ab_noise
import ab_report
import ab_stats
import ab_workloads
import pytest
from ab_stats import EQUAL, FASTER, OPEN, SLOWER, Plan


def table(sd=0.05, df=10, tokens=200):
    return {("server", "decode@30000"): {"sd_percent": sd, "df": df, "decode_tokens": tokens, "source": "t"}}


def test_dumps_and_load_roundtrip(tmp_path):
    t = table()
    path = tmp_path / "noise.toml"
    path.write_text(ab_noise.dumps(t))
    assert ab_noise.load(path) == t
    assert ab_noise.load(tmp_path / "missing.toml") == {}


def test_entries_with_few_runs_or_other_decode_length_are_not_used():
    assert ab_noise.noise_for(table(df=3), ["decode@30000"], {30000: 200}) == {}
    assert ab_noise.noise_for(table(), ["decode@30000"], {30000: 400}) == {}
    assert ab_noise.noise_for(table(), ["decode@30000"], {30000: 200}) == {"decode@30000": (0.05, 10)}
    assert ab_noise.noise_for(table(), ["decode@2000"], {2000: 200}) == {}


def test_known_noise_lets_one_round_decide():
    noise = {"decode@30000": (0.1, 10)}
    rounds = [{"A": {"decode@30000": 100.0}, "B": {"decode@30000": 90.0}}]
    assert ab.all_decided(rounds, Plan(), noise=noise)
    assert not ab.all_decided(rounds, Plan(), noise=None)  # unknown noise: two rounds at least
    k, look, _ = ab_report.decide(rounds, Plan(), noise)["decode@30000"]
    assert (k, look.decision) == (1, SLOWER)


def test_known_small_noise_calls_an_aa_round_equal_after_one_round():
    plan = ab_stats.with_noise(Plan(), 0.05, 10)
    k, look = ab_stats.first_decision([0.03], plan)
    assert (k, look.decision) == (1, EQUAL)


def test_known_noise_cannot_prove_equality_of_a_noisy_metric():
    plan = ab_stats.with_noise(Plan(), 0.5, 10)
    k, look = ab_stats.first_decision([0.1], plan)
    assert k is None and look.decision == OPEN


def draw_table_then_run(rng, true_sd, history, mean, plan):
    """Noise estimated from `history` earlier A/A runs, then a run that looks after every round."""
    old = [rng.gauss(0, true_sd) for _ in range(history)]
    m = sum(old) / history
    sd = (sum((x - m) ** 2 for x in old) / (history - 1)) ** 0.5
    own = ab_stats.with_noise(plan, sd, history - 1)
    diffs = [rng.gauss(mean, true_sd) for _ in range(plan.max_rounds)]
    k, look = ab_stats.first_decision(diffs, own)
    return look.decision if k else OPEN


def test_false_positive_rate_with_a_noisy_table_estimate_stays_below_alpha():
    """10,000 A/A runs, each with its own table value from only 5 earlier runs (the error of the estimate is in it)."""
    rng, plan = random.Random(7), Plan()
    calls = [draw_table_then_run(rng, 0.3, 5, 0.0, plan) for _ in range(10_000)]
    assert (calls.count(FASTER) + calls.count(SLOWER)) / 10_000 <= plan.alpha


def test_false_equivalence_rate_at_the_margin_with_a_noisy_table_stays_below_alpha():
    rng, plan = random.Random(8), Plan()
    calls = [draw_table_then_run(rng, 0.15, 5, plan.margin, plan) for _ in range(10_000)]
    assert calls.count(EQUAL) / 10_000 <= plan.alpha


def test_clear_effect_is_found_after_one_round_in_almost_all_runs():
    rng, plan = random.Random(9), Plan()
    found = 0
    for _ in range(1000):
        own = ab_stats.with_noise(plan, 0.3, 9)
        k, look = ab_stats.first_decision([rng.gauss(-6.0, 0.3)], own)
        found += k == 1 and look.decision == SLOWER
    assert found == 1000


def test_propose_adds_rounds_to_the_old_estimate():
    new = ab_noise.propose(table(sd=0.2, df=10), {"decode@30000": [0.0, 0.4, 0.2]}, {30000: 200}, "run2")
    entry = new[("server", "decode@30000")]
    assert entry["df"] == 12
    assert 0.2 <= entry["sd_percent"] <= 0.21
    assert entry["source"] == "t; run2"


def test_propose_starts_entries_and_drops_a_stable_effect():
    new = ab_noise.propose({}, {"prefill@2000": [-6.0, -6.2, -5.8], "decode-code": [1.0]}, {2000: 400}, "r")
    assert new[("server", "prefill@2000")]["sd_percent"] == pytest.approx(0.2)  # the -6 % effect is not noise
    assert ("bench-set", "decode-code") not in new  # one round gives no variance
    assert new[("server", "prefill@2000")]["decode_tokens"] == 400


def test_propose_does_not_mix_decode_lengths():
    new = ab_noise.propose(table(sd=5.0, df=10, tokens=200), {"decode@30000": [0.0, 0.2]}, {30000: 400}, "r")
    entry = new[("server", "decode@30000")]
    assert entry["df"] == 1 and entry["source"] == "r"


def test_offer_never_writes_the_table_without_an_answer(tmp_path, monkeypatch):
    noise_file = tmp_path / "noise.toml"
    monkeypatch.setattr(ab, "ROOT", tmp_path)
    args = ab.parse_args(["--a", "x", "--b", "y", "--noise", "noise.toml", "--decode-tokens", "200"])
    rounds = [
        {"A": {"decode@1": 100.0}, "B": {"decode@1": 100.0 + d}, "sha": {}, "order": "AB", "seconds": 1}
        for d in (0.1, -0.1, 0.0)
    ]
    ab.offer_noise({"rounds": rounds}, {}, args, {}, tmp_path / "run")
    assert not noise_file.exists()
    proposal = (tmp_path / "run-noise.toml").read_text()
    assert '[server."decode@1"]' in proposal and "df = 2" in proposal


def test_offer_writes_after_a_yes_at_the_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "ROOT", tmp_path)
    monkeypatch.setattr("sys.stdin", type("Tty", (), {"isatty": lambda self: True})())
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    args = ab.parse_args(["--a", "x", "--b", "y", "--noise", "noise.toml"])
    rounds = [
        {"A": {"decode@1": 100.0}, "B": {"decode@1": 100.0 + d}, "sha": {}, "order": "AB", "seconds": 1}
        for d in (0.1, -0.1)
    ]
    ab.offer_noise({"rounds": rounds}, {}, args, {}, tmp_path / "run")
    assert (tmp_path / "noise.toml").exists()


def test_noise_entries_are_tied_to_target_and_ctx_in_the_metric_name():
    t = {("server", "decode@2000/ctx262144"): {"sd_percent": 0.1, "df": 8, "decode_tokens": 400, "source": "t"}}
    assert ab_noise.target_of("decode@2000/ctx262144") == 2000
    assert ab_noise.noise_for(t, ["decode@2000/ctx262144"], {2000: 400}) == {"decode@2000/ctx262144": (0.1, 8)}
    assert ab_noise.noise_for(t, ["decode@2000/ctx140000"], {2000: 400}) == {}
    assert ab_noise.noise_for(t, ["decode@2000/ctx262144"], {2000: 200}) == {}


def test_parse_targets_with_decode_tokens_per_prompt():
    assert ab.parse_targets("2000:400,130000", 200) == [(2000, 400), (130000, 200)]
    with pytest.raises(SystemExit):
        ab.parse_targets("2k", 200)


def test_ctx_must_hold_the_longest_prompt_and_its_answer():
    ok = ab_workloads.jobs_for(["long", "set"], [], [])
    ab_workloads.check_ctx(ok)
    bad = ab_workloads.jobs_for(["server"], [(130000, 200)], [130100])
    with pytest.raises(SystemExit):
        ab_workloads.check_ctx(bad)


def test_metric_names_carry_the_ctx_only_with_several_values():
    one = ab_workloads.metric_names(ab_workloads.jobs_for(["mid"], [], []))
    assert one == ["prefill@30000", "decode@30000"]
    two = ab_workloads.metric_names(ab_workloads.jobs_for(["ctx"], [], []))
    assert "decode@2000/ctx262144" in two and "prefill@2000:write/ctx140000" in two
    assert ab_plan_matches("decode-code/ctx140000", "decode")


def ab_plan_matches(metric, selector):
    import ab_plan

    return ab_plan.matches(metric, selector)
