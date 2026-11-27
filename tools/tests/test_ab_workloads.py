"""Named workloads of scripts/ab.py (docs/contribute/measurement.md): values, server-start groups, metric names."""

import ab_workloads as w
import pytest


def test_the_named_values_are_those_of_the_policy():
    assert w.NAMED["short"].requests == (w.Request("sum", 2000, 400), w.Request("write", 2000, 400))
    assert w.NAMED["short"].ctxs == (140000,)
    assert w.NAMED["ctx"].ctxs == (140000, 262144)
    assert w.NAMED["mid"].requests == (w.Request("sum", 30000, 200),) and w.NAMED["mid"].ctxs == (140000,)
    assert w.NAMED["long"].requests == (w.Request("sum", 130000, 200),)
    assert w.NAMED["set"].bench_set and w.NAMED["set"].ctxs == (32768,)
    huge = w.NAMED["huge"]
    assert [r.tokens for r in huge.requests] == [512000, 1000000] and huge.ctxs == (1048576,)
    assert huge.env == {"TONA_SEQS": "1"}


def test_the_prose_workloads_name_their_requests_by_kind():
    assert w.NAMED["prose"].requests == (w.Request("prose", 2000, 400), w.Request("prose", 16000, 400))
    assert w.NAMED["longdoc"].requests == (w.Request("longdoc", 65536, 200),)
    jobs = w.jobs_for(["longdoc", "prose"], [], [])
    assert [j.ctx for j in jobs] == [32768, 98304]
    assert w.metric_names(jobs)[0] == "prefill@2000:prose"


def test_workloads_with_the_same_ctx_share_one_server_start_in_policy_order():
    jobs = w.jobs_for(["long", "short", "mid"], [], [])
    assert len(jobs) == 1 and jobs[0].ctx == 140000
    assert [r.tokens for r in jobs[0].requests] == [2000, 2000, 30000, 130000]  # short, mid, long


def test_different_ctx_or_environment_start_separately():
    jobs = w.jobs_for(["short", "set", "huge"], [], [])
    assert [(j.ctx, j.env) for j in jobs] == [(140000, {}), (32768, {}), (1048576, {"TONA_SEQS": "1"})]


def test_ctx_workload_runs_short_at_both_values_once_even_with_short_selected():
    jobs = w.jobs_for(["short", "ctx"], [], [])
    assert [j.ctx for j in jobs] == [140000, 262144]
    assert len(jobs[0].requests) == 2


def test_metric_names():
    assert w.metric_names(w.jobs_for(["short"], [], [])) == [
        "prefill@2000",
        "decode@2000",
        "prefill@2000:write",
        "decode@2000:write",
    ]
    assert w.metric_names(w.jobs_for(["set"], [], [])) == ["decode-code", "decode-text"]
    names = w.metric_names(w.jobs_for(["short", "ctx"], [], []))
    assert names.count("decode@2000/ctx262144") == 1 and "decode@2000/ctx140000" in names


def test_unknown_workload_is_refused():
    with pytest.raises(SystemExit, match="unknown workload"):
        w.jobs_for(["tiny"], [], [])


def test_custom_server_workload_uses_targets_and_ctx():
    jobs = w.jobs_for(["server"], [(30000, 200), (2000, 400)], [140000])
    assert [r.tokens for r in jobs[0].requests] == [30000, 2000]
    assert w.tokens_by_target(jobs) == {30000: 200, 2000: 400}


def test_ctx_check_and_description():
    with pytest.raises(SystemExit):
        w.check_ctx(w.jobs_for(["server"], [(130000, 200)], [130100]))
    text = w.describe(w.jobs_for(["short", "huge"], [], []))
    assert "2000:400" in text and "TONA_SEQS=1" in text and "--ctx 1048576" in text
