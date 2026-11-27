"""Protection against outliers in one-round decisions: a round outside the plan or an outlier needs another round."""

import random

import ab
import ab_report
import ab_stats
from ab_stats import FASTER, OPEN, SLOWER, Plan

NOISE = {"decode@130000": (0.4, 9)}
EXPECT = [("decode", -8.0, -4.0)]


def rounds_of(diffs, metric="decode@130000"):
    return [{"A": {metric: 100.0}, "B": {metric: 100.0 + d}} for d in diffs]


def test_a_one_round_decision_inside_the_plan_is_final():
    k, look, _ = ab_report.decide(rounds_of([-6.0]), Plan(), NOISE, EXPECT)["decode@130000"]
    assert (k, look.decision) == (1, SLOWER)


def test_a_one_round_decision_outside_the_plan_needs_another_round():
    one = ab_report.decide(rounds_of([-23.7]), Plan(), NOISE, EXPECT)["decode@130000"]
    assert one[0] is None
    assert not ab.all_decided(rounds_of([-23.7]), Plan(), noise=NOISE, expect=EXPECT)
    k, look, _ = ab_report.decide(rounds_of([-23.7, -23.5]), Plan(), NOISE, EXPECT)["decode@130000"]
    assert (k, look.decision) == (2, SLOWER)  # consistent twice: decided, the plan check still calls it a finding


def test_an_off_plan_first_round_followed_by_an_in_plan_round_is_decided_with_two():
    k, look, _ = ab_report.decide(rounds_of([-20.0, -6.0]), Plan(), NOISE, EXPECT)["decode@130000"]
    assert k == 2 and look.decision == SLOWER


def test_flags_mark_off_plan_and_outlier_rounds_and_drop_nothing():
    flags = ab_report.round_flags([-6.0, -6.1, -20.0, -5.9], 0.4, (-8.0, -4.0))
    assert flags == {3: "outlier"}
    assert ab_report.round_flags([-23.7, -6.0], 0.4, (-8.0, -4.0)) == {1: "off plan"}
    assert ab_report.round_flags([-6.0, -6.1], None, (-8.0, -4.0)) == {}


def test_an_outlier_round_cannot_be_the_deciding_one():
    # without the gate round 2 would decide "faster" (mean 1.65, clear against the noise 0.3)
    plan = ab_stats.with_noise(Plan(), 0.3, 9)
    assert ab_stats.first_decision([0.3, 3.0], plan)[0] == 2
    flags = ab_report.round_flags([0.3, 3.0, 3.0, 3.0], 0.3, None)

    def gate(n):
        return n not in flags

    k, look = ab_stats.first_decision([0.3, 3.0, 3.0, 3.0], plan, gate)
    assert k == 4 and look.decision == FASTER  # the outlier rounds 2 and 3 are in the data, round 4 decides


def test_the_report_marks_the_flagged_rounds():
    rounds = [{**r, "order": "AB", "seconds": 1, "sha": {}} for r in rounds_of([-6.0, -6.1, -20.0])]
    text = "\n".join(ab_report.round_rows(rounds, ab_report.flags_of(rounds, NOISE, EXPECT)))
    assert "-20.00 % (outlier)" in text and "-6.00 %" in text and "(off plan)" not in text


def run_decisions(diffs_of_run, noise_sd, expect, runs, seed):
    rng, plan = random.Random(seed), Plan()
    out = []
    for _ in range(runs):
        history = [rng.gauss(0, noise_sd) for _ in range(5)]
        mean = sum(history) / 5
        sd = (sum((x - mean) ** 2 for x in history) / 4) ** 0.5
        diffs = diffs_of_run(rng)
        metric = "decode@130000"
        rounds = rounds_of(diffs, metric)
        k, look, _ = ab_report.decide(rounds, plan, {metric: (sd, 4)}, expect)[metric]
        out.append(look.decision if k else OPEN)
    return out


def test_false_positive_rate_with_the_protection_stays_below_alpha():
    plan = Plan()
    calls = run_decisions(lambda r: [r.gauss(0, 0.3) for _ in range(6)], 0.3, [("decode", -1.0, 1.0)], 5000, 11)
    assert (calls.count(FASTER) + calls.count(SLOWER)) / 5000 <= plan.alpha


def test_a_spike_in_round_one_never_decides_alone():
    def run(r):
        return [-20.0] + [r.gauss(-6.0, 0.3) for _ in range(5)]

    calls = run_decisions(run, 0.3, EXPECT, 300, 12)
    assert calls.count(SLOWER) == 300  # decided, but at round 2 or later (see the tests above), never at round 1


def test_a_spike_in_a_later_round_is_flagged_in_almost_all_runs():
    rng, flagged = random.Random(13), 0
    for _ in range(1000):
        diffs = [rng.gauss(-6.0, 0.3) for _ in range(3)] + [-6.0 - 3.0]
        flagged += 4 in ab_report.round_flags(diffs, 0.3, (-8.0, -4.0))
    assert flagged == 1000


def test_normal_rounds_are_rarely_flagged():
    rng, flagged = random.Random(14), 0
    for _ in range(5000):
        diffs = [rng.gauss(-6.0, 0.3) for _ in range(4)]
        flagged += bool(ab_report.round_flags(diffs, 0.3, (-8.0, -4.0)))
    assert flagged / 5000 < 0.01


def test_rounds_continue_after_an_off_plan_first_round():
    def measure(key):
        f = 0.77 if key == "B" else 1.0  # -23 %: far outside the plan -8 .. -4
        return {"decode@130000": 100 * f}, {"130000": "s"}

    rounds = []
    ab.run_rounds(measure, Plan(max_rounds=3), rounds, stop=ab.Stop(None, NOISE, EXPECT))
    assert len(rounds) == 2
