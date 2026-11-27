"""Tests of the measurement harness; no GPU needed."""

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path

from ibench import client, report, runner, stats, suite
from ibench.sysmon import SysMon

from tests.mock import Mock

SUITE = """
name = "test"
[defaults]
runs = 3
warmup = 1
[[case]]
name = "short"
prompt = "Write fib."
expect = 'def\\s+fib'
min_decode_tokens = 8
[[case]]
name = "needle"
context_chars = 5000
needle = "kestrel-4711"
question = "Value?"
cache = "warm"
"""


class TestStats(unittest.TestCase):
    def test_summarize(self):
        s = stats.summarize([3, 1, None, 2])
        self.assertEqual((s["n"], s["median"], s["min"], s["max"]), (3, 2, 1, 3))
        self.assertAlmostEqual(s["stdev"], 1.0)
        self.assertIsNone(stats.summarize([None]))

    def test_percentile(self):
        self.assertEqual(stats.percentile([1, 2, 3, 4, 5], 0.5), 3)
        self.assertAlmostEqual(stats.percentile([0, 10], 0.99), 9.9)


class TestClient(unittest.TestCase):
    def setUp(self):
        self.mock = Mock(first_s=0.2, n=11, gap_s=0.02)

    def tearDown(self):
        self.mock.close()

    def test_openai_timing(self):
        r = client.stream(self.mock.url, "openai", {"model": "m", "messages": []})
        self.assertTrue(r.ok, r.error)
        self.assertAlmostEqual(r.t_first - r.t_start, 0.2, delta=0.05)
        self.assertEqual(len(r.chunk_times), 11)
        self.assertEqual((r.prompt_tokens, r.completion_tokens, r.finish), (1000, 11, "stop"))
        self.assertEqual(r.text, self.mock.text)
        self.assertTrue(self.mock.requests[0]["stream"])
        self.assertEqual(self.mock.requests[0]["stream_options"], {"include_usage": True})

    def test_anthropic(self):
        self.mock.cached_tokens = 400
        r = client.stream(self.mock.url, "anthropic", {"model": "m", "messages": [], "max_tokens": 10})
        self.assertTrue(r.ok, r.error)
        self.assertEqual((r.prompt_tokens, r.cached_tokens, r.completion_tokens), (1000, 400, 11))
        self.assertEqual(r.finish, "end_turn")
        self.assertAlmostEqual(r.t_first - r.t_start, 0.2, delta=0.05)

    def test_http_error(self):
        r = client.stream("http://127.0.0.1:1", "openai", {})
        self.assertFalse(r.ok)
        self.assertIn("Error", r.error)


class TestMetrics(unittest.TestCase):
    def test_decode_and_mtp(self):
        mock = Mock(first_s=0.1, n=40, gap_s=0.01, per_chunk=4)  # 10 chunks of 4 tokens
        try:
            r = client.stream(mock.url, "openai", {"model": "m", "messages": []})
        finally:
            mock.close()
        case = suite.Case(name="x", prompt="p", min_decode_tokens=8, expect="def")
        m = runner.metrics(r, case)
        # 39 tokens over 9 gaps of 10 ms
        self.assertAlmostEqual(m["decode_tps_client"], 39 / 0.09, delta=39 / 0.09 * 0.35)
        self.assertAlmostEqual(m["prefill_tps_client"], 1000 / 0.1, delta=1000 / 0.1 * 0.35)
        self.assertEqual((m["prefill_tps_server"], m["decode_tps_server"]), (800.0, 50.0))
        self.assertAlmostEqual(m["tokens_per_pass"], 40 / 34)
        self.assertAlmostEqual(m["draft_accept_rate"], 6 / 9)
        self.assertTrue(m["expect_ok"])

    def test_short_answer_has_no_client_decode(self):
        r = client.Result(
            ok=True,
            t_start=0,
            t_first=0.1,
            t_last=0.2,
            t_end=0.2,
            chunk_times=[0.1, 0.2],
            completion_tokens=2,
            prompt_tokens=10,
        )
        m = runner.metrics(r, suite.Case(name="x", prompt="p"))
        self.assertNotIn("decode_tps_client", m)


class TestSuite(unittest.TestCase):
    def load(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(SUITE)
        return suite.load(f.name)

    def test_load_and_defaults(self):
        meta, cases = self.load()
        self.assertEqual(meta["name"], "test")
        self.assertEqual([c.runs for c in cases], [3, 3])

    def test_cold_nonce_breaks_prefix(self):
        _, cases = self.load()
        a = suite.bodies(cases[0], "openai", "m", 0, "aaa", {})[0]
        b = suite.bodies(cases[0], "openai", "m", 1, "bbb", {})[0]
        self.assertTrue(a["messages"][0]["content"].startswith("[run aaa]"))
        self.assertNotEqual(a["messages"][0]["content"][:12], b["messages"][0]["content"][:12])

    def test_warm_shares_prefix(self):
        _, cases = self.load()
        a = suite.bodies(cases[1], "openai", "m", 0, None, {})[0]["messages"]
        b = suite.bodies(cases[1], "openai", "m", 1, None, {})[0]["messages"]
        self.assertEqual(a[0], b[0])
        self.assertNotEqual(a[1]["content"], b[1]["content"])
        self.assertTrue(b[1]["content"].startswith(a[1]["content"][:-3]))
        self.assertIn("kestrel-4711", a[1]["content"])

    def test_context_is_deterministic(self):
        c1 = suite.context(5000, "n1", 0.5)
        self.assertEqual(c1, suite.context(5000, "n1", 0.5))
        self.assertLessEqual(len(c1), 5000 + 80)
        self.assertTrue(c1.endswith("\n"))

    def test_rejects_unknown_field(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write('[[case]]\nname = "x"\nprompt = "p"\nmax_token = 5\n')
        with self.assertRaises(ValueError):
            suite.load(f.name)

    def test_anthropic_trace_nonce(self):
        with tempfile.TemporaryDirectory() as d:
            tr = Path(d) / "t.jsonl"
            tr.write_text(
                json.dumps(
                    {
                        "api": "anthropic",
                        "body": {
                            "model": "x",
                            "system": [{"type": "text", "text": "S"}],
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    }
                )
                + "\n"
            )
            case = suite.Case(name="t", trace=str(tr))
            b = suite.bodies(case, "anthropic", "m", 0, "nn", {})
            self.assertEqual(b[0]["system"][0]["text"], "[run nn]\n")
            self.assertEqual(b[0]["model"], "m")
            with self.assertRaises(ValueError):
                suite.bodies(case, "openai", "m", 0, None, {})


class TestEndToEnd(unittest.TestCase):
    def test_run_and_report(self):
        mock = Mock(first_s=0.05, n=12, gap_s=0.005)
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
                f.write(SUITE)
            meta, cases = suite.load(f.name)
            target = {"name": "mock", "url": mock.url, "api": "openai", "model": "m", "extra": {"seed": 1}}
            mon = SysMon(0.05).start()
            results = [runner.run_case(c, target, mon) for c in cases]
            mon.stop()
        finally:
            mock.close()
        self.assertEqual(len(mock.requests), 8)  # 2 cases x (1 warm-up run + 3 runs)
        self.assertEqual(mock.requests[0]["seed"], 1)
        s = results[0]["summary"]["steps"][0]
        self.assertEqual((s["n"], s["ok"], s["expect_pass"], s["expect_n"]), (3, 3, 3, 3))
        self.assertEqual(s["metrics"]["completion_tokens"]["median"], 12)
        env = {
            "time": "2026-10-23T10:00:00+02:00",
            "target": target,
            "git": {"commit": "abc", "dirty": False},
            "mem_total_gib": 62.0,
            "mem_avail_gib": 50.0,
            "swap_used_mib": 0,
            "warnings": [],
        }
        data = {"suite": meta, "env": env, "cases": results}
        with tempfile.TemporaryDirectory() as d:
            base = report.unique_path(Path(d), "x")
            jp, mp = report.write(data, base)
            self.assertEqual(report.unique_path(Path(d), "x").name, "x-2")
            md = mp.read_text()
            self.assertIn("| short | 3/3 | 3/3 |", md)
            loaded = json.loads(jp.read_text())
            self.assertIn("B/A", report.compare(loaded, loaded))


class TestProxy(unittest.TestCase):
    def test_records_and_streams(self):
        from ibench import proxy

        mock = Mock(first_s=0.05, n=5, gap_s=0.01)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "t.jsonl"
            srv = proxy.serve("127.0.0.1:0", mock.url, str(out))
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try:
                url = f"http://127.0.0.1:{srv.server_address[1]}"
                r = client.stream(url, "openai", {"model": "m", "messages": [{"role": "user", "content": "x"}]})
                self.assertTrue(r.ok, r.error)
                self.assertEqual(r.completion_tokens, 5)
                self.assertAlmostEqual(r.t_first - r.t_start, 0.05, delta=0.04)
                conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
                conn.request("GET", "/v1/models")
                self.assertEqual(conn.getresponse().status, 200)
                lines = [json.loads(x) for x in out.read_text().splitlines()]
                self.assertEqual(len(lines), 1)  # GET is not recorded
                self.assertEqual(lines[0]["api"], "openai")
                self.assertEqual(lines[0]["body"]["messages"][0]["content"], "x")
            finally:
                srv.shutdown()
                srv.server_close()
                mock.close()


class TestSysMon(unittest.TestCase):
    def test_window(self):
        mon = SysMon(0.02).start()
        import time

        t0 = time.perf_counter()
        time.sleep(0.1)
        t1 = time.perf_counter()
        mon.stop()
        w = mon.window(t0, t1)
        self.assertGreaterEqual(w["samples"], 2)
        self.assertIn("ram_avail_min_mib", w)


if __name__ == "__main__":
    unittest.main()
