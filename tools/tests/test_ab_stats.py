"""The sequential decision rule of scripts/ab.py: stops early on clear effects, keeps its error budget."""

import random

import ab_stats
import pytest
from ab_stats import EQUAL, FASTER, OPEN, SLOWER, Plan, first_decision, summarize


def test_t_quantile_matches_tables():
    assert ab_stats.t_quantile(0.025, 10) == pytest.approx(2.2281, abs=1e-4)
    assert ab_stats.t_quantile(0.05, 1) == pytest.approx(6.3138, abs=1e-4)
    assert ab_stats.t_quantile(0.005, 30) == pytest.approx(2.7500, abs=1e-4)


def test_t_sf_is_symmetric_around_the_median():
    assert ab_stats.t_sf(0.0, 7) == 0.5
    assert ab_stats.t_sf(1e-9, 3) == pytest.approx(0.5, abs=1e-6)


def test_levels_sum_to_alpha():
    plan = Plan()
    total = sum(ab_stats.level(plan, k) for k in range(plan.min_rounds, plan.max_rounds + 1))
    assert total == pytest.approx(plan.alpha)


def test_clear_effect_stops_after_two_rounds():
    k, look = first_decision([5.1, 4.9, 5.0, 5.2], Plan())
    assert (k, look.decision) == (2, FASTER)
    k, look = first_decision([-3.0, -3.2], Plan())
    assert (k, look.decision) == (2, SLOWER)


def test_no_decision_before_the_minimum_rounds():
    assert first_decision([50.0], Plan())[0] is None


def test_no_effect_becomes_equal():
    k, look = first_decision([0.05, -0.08, 0.02, 0.0, -0.03, 0.06], Plan())
    assert look.decision == EQUAL and k is not None


def test_strong_noise_stays_open_until_the_limit():
    k, look = first_decision([4.0, -3.5, 2.0, -4.2, 3.1, -2.2], Plan())
    assert k is None and look.decision == OPEN and look.n == 6


def test_decision_is_final_even_if_later_rounds_disagree():
    k, look = first_decision([5.0, 5.1, -9.0, -9.0, -9.0, -9.0], Plan())
    assert (k, look.decision) == (2, FASTER)


def test_small_real_gain_needs_more_rounds_than_a_big_one():
    plan = Plan()
    small = first_decision([0.9, 1.1, 1.0, 0.95, 1.05, 1.0], plan)[0]
    big = first_decision([9.0, 11.0], plan)[0]
    assert big == 2 and (small is None or small > 2)


def simulate(mean: float, sd: float, runs: int, plan: Plan, seed: int) -> dict:
    rng = random.Random(seed)
    counts = {FASTER: 0, SLOWER: 0, EQUAL: 0, OPEN: 0}
    for _ in range(runs):
        diffs = [rng.gauss(mean, sd) for _ in range(plan.max_rounds)]
        k, look = first_decision(diffs, plan)
        counts[look.decision if k else OPEN] += 1
    return counts


def test_false_positive_rate_of_aa_runs_stays_below_alpha():
    """10,000 runs without a difference, noise as assumed by the prior: faster/slower in at most alpha of them."""
    plan = Plan()
    counts = simulate(0.0, plan.prior_sd, 10_000, plan, seed=1)
    assert (counts[FASTER] + counts[SLOWER]) / 10_000 <= plan.alpha


def test_false_positive_rate_with_less_noise_than_assumed():
    plan = Plan(prior_sd=0.5)
    counts = simulate(0.0, 0.25, 10_000, plan, seed=2)
    assert (counts[FASTER] + counts[SLOWER]) / 10_000 <= plan.alpha


def test_false_equivalence_rate_at_the_margin_stays_below_alpha():
    """A true difference exactly at the margin must be called 'equal' in at most alpha of the runs."""
    plan = Plan(prior_sd=0.25)
    counts = simulate(plan.margin, 0.25, 10_000, plan, seed=3)
    assert counts[EQUAL] / 10_000 <= plan.alpha


def test_aa_with_low_noise_is_called_equal_early():
    plan = Plan(prior_sd=0.25)
    counts = simulate(0.0, 0.15, 2_000, plan, seed=4)
    assert counts[EQUAL] >= 1_980


def test_summarize_reports_interval_around_the_mean():
    look = summarize([1.0, 2.0, 3.0], Plan())
    assert look.mean == pytest.approx(2.0)
    assert look.ci_diff[0] < 2.0 < look.ci_diff[1]
    assert look.ci_equiv[0] > look.ci_diff[0]  # TOST interval is narrower (one-sided levels)


def test_a_real_difference_inside_the_margin_is_marked():
    plan = ab_stats.with_noise(Plan(), 0.02, 20)
    look = summarize([0.11], plan)
    assert look.decision == FASTER and look.inside_margin
    assert not summarize([5.0], plan).inside_margin
