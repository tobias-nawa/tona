"""The R8 pressure series: the summary against the expectation and the refusal of the service port."""

import json
import sys

import pytest
import r8_pressure


def row(decode=70.0, evicted=100, episode=50, pressured=True, **extra):
    return {
        "pair": 1,
        "target": 30000,
        "pressured": pressured,
        "decode": decode,
        "evicted_ms": evicted,
        "longest_episode_ms": episode,
        **extra,
    }


def write(tmp_path, rows):
    path = tmp_path / "r8.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_a_clean_series_meets_the_expectation(tmp_path):
    assert r8_pressure.summarize(write(tmp_path, [row(), row(pressured=False), row()])) == 0


def test_a_failed_request_misses(tmp_path):
    assert (
        r8_pressure.summarize(write(tmp_path, [row(), {"pair": 1, "target": 130000, "pressured": True, "error": "x"}]))
        == 1
    )


def test_four_slow_decodes_miss_and_three_pass(tmp_path):
    assert r8_pressure.summarize(write(tmp_path, [row(decode=40.0)] * 3 + [row()])) == 0
    assert r8_pressure.summarize(write(tmp_path, [row(decode=40.0)] * 4 + [row()])) == 1


def test_eviction_time_and_episode_limits(tmp_path):
    assert r8_pressure.summarize(write(tmp_path, [row(evicted=5000)])) == 1
    assert r8_pressure.summarize(write(tmp_path, [row(episode=3000)])) == 1


def test_unpressured_requests_do_not_count(tmp_path):
    assert (
        r8_pressure.summarize(write(tmp_path, [row(), row(pressured=False, decode=10.0, evicted=99999, episode=99999)]))
        == 0
    )


@pytest.mark.parametrize("port", [8662, 8097])
def test_the_port_of_the_service_is_refused(monkeypatch, port):
    monkeypatch.setattr(sys, "argv", ["r8_pressure.py", "--build", "x", "--pressure-files", "y", "--port", str(port)])
    with pytest.raises(SystemExit):
        r8_pressure.parse_args()
