#!/usr/bin/env python3
"""Replays stored agent requests at their original arrival times against a running server and records what the clock
hold (TONA_CLOCK_HOLD) changes: time to first token, prefill rate, and the GPU energy over the run.

The request files are Anthropic /v1/messages bodies named HH:MM:SS-NNNN.json (the arrival time is in the name). They are
data of the owner of the machine: pass a directory, nothing from it is copied anywhere. Output: one JSON file.
Usage: scripts/replay_hold.py <dumps dir> <first> <count> --port 8099 --out result.json [--hwmon DIR]
"""

import argparse
import http.client
import json
import threading
import time
from pathlib import Path


def arrival_seconds(p: Path) -> int:
    h, m, s = p.name.split("-")[0].split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


class Sampler(threading.Thread):
    """hwmon power (W) and clock (MHz) every 100 ms; the energy is the sum of power times interval."""

    def __init__(self, hwmon: Path):
        super().__init__(daemon=True)
        self.hwmon, self.rows, self.stop = hwmon, [], threading.Event()

    def num(self, name: str) -> float:
        return float((self.hwmon / name).read_text())

    def run(self):
        while not self.stop.is_set():
            self.rows.append((time.monotonic(), self.num("power1_average") / 1e6, self.num("freq1_input") / 1e6))
            time.sleep(0.1)


def send(port: int, body: dict, out: dict):
    """One request, streaming; out gets ttft_s (first content delta), total_s, and the event count."""
    body = {**body, "stream": True, "max_tokens": min(body.get("max_tokens", 32), 32)}
    t0 = time.monotonic()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    c.request(
        "POST",
        "/v1/messages",
        json.dumps(body),
        {"content-type": "application/json", "x-api-key": "x", "anthropic-version": "2023-06-01"},
    )
    r = c.getresponse()
    out["status"] = r.status
    for line in r:
        if line.startswith(b"data:") and b"content_block_delta" in line and "ttft_s" not in out:
            out["ttft_s"] = time.monotonic() - t0
    out["total_s"] = time.monotonic() - t0
    out["end"] = time.monotonic()
    c.close()


def replay(port: int, files: list, t_start: float) -> list:
    """Sends files[1:] at their original arrival offsets (open loop); returns the result dicts once all are answered."""
    base = arrival_seconds(files[0])
    results, threads = [], []
    for f in files:
        delay = t_start + arrival_seconds(f) - base - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        res = {"file_t": arrival_seconds(f) - base, "sent": time.monotonic() - t_start}
        th = threading.Thread(target=send, args=(port, json.loads(f.read_text()), res))
        th.start()
        results.append(res), threads.append(th)
    for th in threads:
        th.join()
    return results


def power_summary(rows: list, results: list, t_start: float) -> dict:
    """Energy over the run and mean power while no request runs (the pauses) against while one runs."""
    busy = [(r["sent"] + t_start, r["end"]) for r in results]
    idle_w, busy_w = [], []
    for t, w, _ in rows:
        (busy_w if any(s <= t <= e for s, e in busy) else idle_w).append(w)
    return {
        "wall_s": rows[-1][0] - rows[0][0],
        "energy_j": sum(w * 0.1 for _, w, _ in rows),
        "idle_w_mean": sum(idle_w) / max(len(idle_w), 1),
        "idle_samples": len(idle_w),
        "busy_w_mean": sum(busy_w) / max(len(busy_w), 1),
    }


def server_status(port: int) -> dict:
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("GET", "/status")
    return json.loads(c.getresponse().read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps")
    ap.add_argument("first", type=int)
    ap.add_argument("count", type=int)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hwmon", default="/sys/class/drm/card1/device/hwmon/hwmon3")
    a = ap.parse_args()
    files = sorted(Path(a.dumps).glob("2*.json"), key=lambda p: p.name)[a.first : a.first + a.count + 1]
    # The first request only fills the prompt cache and is not measured
    warm = {}
    send(a.port, json.loads(files[0].read_text()), warm)
    if warm.get("status") != 200:
        raise SystemExit(f"first request answered {warm.get('status')}: the server is not ready")
    time.sleep(5)
    smp = Sampler(Path(a.hwmon))
    smp.start()
    t_start = time.monotonic()
    results = replay(a.port, files[1:], t_start)
    time.sleep(2)
    smp.stop.set()
    smp.join()
    if any(r.get("status") != 200 for r in results):
        raise SystemExit("a replayed request failed: " + str([r.get("status") for r in results]))
    status = server_status(a.port)
    out = {
        "requests": [{k: v for k, v in r.items() if k != "end"} for r in results],
        "recent": status.get("recent", [])[-len(results) :],
        **power_summary(smp.rows, results, t_start),
        "clock_hold": status.get("engine", {}).get("clock_hold"),
    }
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
