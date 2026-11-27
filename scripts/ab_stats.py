"""Sequential decision rule of scripts/ab.py: when is a paired A/B measurement clear enough to stop?

Looking at the data after every round and stopping at the first clear result inflates the false-positive rate
(Armitage, McPherson and Rowe 1969). The remedy is a group-sequential test with a fixed error budget: the total
alpha is spread over the planned looks by a spending function (Lan and DeMets 1983, Biometrika 70:659-663).
Here the Pocock-type function a(t) = alpha * ln(1 + (e - 1) t) is used (Lan and DeMets 1983, eq. 2.1): it spends
evenly, which suits few, expensive rounds (a clear effect must stop after 2 of 6), where O'Brien-Fleming would
demand an effect of several sigma before round 4. The level of look k is the increment a(t_k) - a(t_k-1), tested
at its own level. The sum of the levels is alpha, so by the union bound the family-wise error stays below alpha
whatever the correlation between the looks (conservative against the exact recursion of Jennison and Turnbull 2000,
chapter 7, which needs numerical integration).

Per metric and round the statistic is the paired difference d = 100 * (B / A - 1) in percent. Two questions:

- difference: two-sided test of mean(d) = 0 at the level of the look; rejected means "B faster" or "B slower".
- equivalence: TOST (Schuirmann 1987, J Pharmacokinet Biopharm 15:657-680): mean(d) is inside +-margin when both
  one-sided tests against -margin and +margin reject, each at the level of the look.

With 2 rounds there is one degree of freedom, and a t-test on it would need a difference of several hundred
sigma at these levels. The variance is therefore a conjugate mix of the sample variance and a prior from earlier
A/A runs (prior_sd with prior_df pseudo degrees of freedom): s2 = (prior_df prior_sd^2 + (n - 1) s^2) / (prior_df
+ n - 1), t distribution with prior_df + n - 1 degrees of freedom. The guarantee then holds if the true round to
round noise is not above prior_sd; tests/test_ab_stats.py simulates both cases.

When the noise of a workload is known from at least five earlier A/A runs (bench/noise.toml, scripts/ab_noise.py),
that sample variance and its degrees of freedom replace the guess, and the first look is allowed after one round:
a pooled-variance t test with the variance of the earlier runs, exact if the noise is unchanged (Student 1908;
Gelman et al., Bayesian Data Analysis, 3rd ed., section 3.3, for the same model with a scaled inverse chi-square
prior). The simulation in the tests draws the table value from ten runs each time, so the error of that
estimate is in it.
"""

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from statistics import mean, variance

FASTER, SLOWER, EQUAL, OPEN = "B faster", "B slower", "equal", "open"


@dataclass(frozen=True)
class Plan:
    alpha: float = 0.05
    margin: float = 0.5  # percent: differences below it are "equal"
    max_rounds: int = 6
    min_rounds: int = 2
    prior_sd: float = 0.5  # percent: round to round noise of the paired difference seen in A/A runs
    prior_df: float = 4.0


def with_noise(plan: Plan, sd: float, df: float) -> Plan:
    """The plan of a metric whose noise is known from earlier runs (scripts/ab_noise.py): that variance with its
    degrees of freedom is the prior, and one round may decide."""
    return replace(plan, prior_sd=sd, prior_df=float(df), min_rounds=1)


def spent(plan: Plan, k: int) -> float:
    """Cumulative alpha after look k (Pocock-type spending, information fraction k / max_rounds)."""
    t = min(k / plan.max_rounds, 1.0)
    return plan.alpha * math.log(1 + (math.e - 1) * t)


def level(plan: Plan, k: int) -> float:
    """Nominal one-sided-total level of look k: what the spending function releases since the last look."""
    previous = spent(plan, k - 1) if k > plan.min_rounds else 0.0
    return spent(plan, k) - previous


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (Lentz's method, Numerical Recipes 6.4)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for num in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + num * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + num / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1) < 1e-14:
            break
    return h


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with df degrees of freedom (t >= 0)."""
    if t < 1e-12:
        return 0.5
    x = df / (df + t * t)
    y = t * t / (df + t * t)  # 1 - x without cancellation
    a, b = df / 2, 0.5
    # I_x(a, b) / 2 is the tail; the continued fraction converges fast on one side of the pivot only
    front = math.exp(a * math.log(x) + b * math.log(y) + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
    if x < (a + 1) / (a + b + 2):
        return 0.5 * front * _betacf(a, b, x) / a
    return 0.5 * (1 - front * _betacf(b, a, y) / b)


@lru_cache(maxsize=4096)
def t_quantile(p: float, df: float) -> float:
    """t such that P(T > t) = p, for 0 < p < 0.5 (bisection; the arguments repeat, so the result is cached)."""
    lo, hi = 0.0, 1.0
    while t_sf(hi, df) > p:
        hi *= 2
        if hi > 1e12:
            break
    for _ in range(80):
        mid = (lo + hi) / 2
        if t_sf(mid, df) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass(frozen=True)
class Look:
    n: int
    mean: float
    se: float
    df: float
    ci_diff: tuple  # two-sided interval at the level of the look
    ci_equiv: tuple  # (1 - 2 level) interval of the TOST
    decision: str
    inside_margin: bool = False  # TOST passed: whatever the decision, the difference is below the margin


def summarize(diffs: list, plan: Plan, k: int | None = None) -> Look:
    """The decision on the paired differences (percent) of the first n rounds, at the level of look n (or k)."""
    n = len(diffs)
    k = n if k is None else k
    m = mean(diffs)
    s2 = variance(diffs) if n > 1 else 0.0
    nu = plan.prior_df + n - 1
    pooled = (plan.prior_df * plan.prior_sd**2 + (n - 1) * s2) / nu
    se = math.sqrt(pooled / n)
    a = max(level(plan, max(k, plan.min_rounds)), 1e-12)
    t_diff, t_equiv = t_quantile(a / 2, nu), t_quantile(a, nu)
    ci_diff = (m - t_diff * se, m + t_diff * se)
    ci_equiv = (m - t_equiv * se, m + t_equiv * se)
    inside = ci_equiv[0] > -plan.margin and ci_equiv[1] < plan.margin
    if ci_diff[0] > 0:
        decision = FASTER
    elif ci_diff[1] < 0:
        decision = SLOWER
    elif inside:
        decision = EQUAL
    else:
        decision = OPEN
    return Look(n, m, se, nu, ci_diff, ci_equiv, decision, inside)


def first_decision(diffs: list, plan: Plan, gate=None) -> tuple:
    """(round, Look) of the first look at which the metric is decided, else (None, Look of all rounds).

    Only looks from min_rounds on count, and a decision is final: later rounds never revoke it. gate(n) -> bool can
    veto a look (scripts/ab_report.py: a round outside the plan or an outlier must be followed by another round);
    a veto removes a look, it never adds one, so the error budget is not touched.
    """
    last = None
    for n in range(max(plan.min_rounds, 1), min(len(diffs), plan.max_rounds) + 1):
        last = summarize(diffs[:n], plan)
        if last.decision != OPEN and (gate is None or gate(n)):
            return n, last
    return None, last
