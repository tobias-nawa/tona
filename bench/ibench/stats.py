"""Statistics of a measurement series."""

import statistics


def summarize(values) -> dict | None:
    """Median, mean, stdev, min, max over all values that are not None."""
    v = [float(x) for x in values if x is not None]
    if not v:
        return None
    return {
        "n": len(v),
        "median": statistics.median(v),
        "mean": statistics.fmean(v),
        "stdev": statistics.stdev(v) if len(v) > 1 else 0.0,
        "min": min(v),
        "max": max(v),
    }


def percentile(values, q: float) -> float | None:
    """Percentile with linear interpolation, q in [0, 1]."""
    v = sorted(values)
    if not v:
        return None
    pos = (len(v) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)
