"""The numbers of the desktop latency sweep without a GPU (tools/desktop_latency.py)."""

import argparse

import desktop_latency as dl


def test_frame_hit_rate_counts_the_budget_inclusive():
    assert dl.frame_hit_rate([1000, 16700, 16701, 50000]) == 0.5
    assert dl.frame_hit_rate([]) is None


def test_phase_summary_uses_only_the_window():
    canary = [(1.0, 100.0), (5.0, 1000.0), (6.0, 20000.0), (12.0, 99999.0)]
    rates = [(4.0, 70.0), (5.5, 80.0), (11.0, 10.0)]
    out = dl.phase_summary(canary, rates, 5.0, 10.0)
    assert out["canary_submissions"] == 2 and out["canary_us_max"] == 20000.0
    assert out["frame_hit_rate"] == 0.5
    assert out["tona_requests"] == 1 and out["tona_tok_s_mean"] == 80.0
    empty = dl.phase_summary([], [], 0.0, 1.0)
    assert empty["canary_us_p50"] is None and empty["frame_hit_rate"] is None


def test_env_must_be_tona_variables():
    assert dl.parse_env(["TONA_FREE_CUS=4"]) == {"TONA_FREE_CUS": "4"}
    try:
        dl.parse_env(["HSA_CU_MASK=0:0-3"])
    except SystemExit:
        return
    raise AssertionError("a non-TONA variable was accepted")


def test_prefill_prompts_differ_per_request(monkeypatch):
    seen = []
    monkeypatch.setattr(
        dl.ab_server,
        "post",
        lambda url, path, body, t: seen.append(body) or {"timings": {"prompt_n": 2000, "prompt_ms": 1000.0}},
    )
    w = dl.Worker("http://x", "code")
    w.request("prefill")
    w.request("prefill")
    heads = [b["messages"][0]["content"].split("\n")[0] for b in seen]
    assert heads[0] != heads[1]
    assert w.rates["prefill"][0][1] == 2000.0
    assert argparse  # the module under test is importable without a GPU
