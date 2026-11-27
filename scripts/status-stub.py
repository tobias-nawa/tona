#!/usr/bin/env python3
"""A stand-in for the engine's /status: serves a recorded document so the control centre can be run, and photographed
(scripts/screenshot-tui.py), without a server and a GPU.

    scripts/status-stub.py [--port 8674] [--file engine/tests/data/client/status-decode.json]

The first running request writes SPEED tokens per second: its out_tokens grows by a fixed step per poll, so the write
rate the client derives is the same on every run (the client polls every 250 ms). Port 8662 is the real server's and
is refused.
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLL_S = 0.25  # the poll interval of the client (engine/client/poller.cpp)
REAL_PORT = 8662


def advance(doc: dict, polls: int, speed: float) -> dict:
    """The document after `polls` polls: the first running request has written speed * POLL_S more tokens per poll."""
    out = json.loads(json.dumps(doc))
    first = out["engine"]["active"][0]
    first["out_tokens"] = first["out_tokens"] + round(speed * POLL_S * polls)
    return out


def make_handler(doc: dict, speed: float):
    polls = [0]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/status":
                self.send_error(404)
                return
            body = json.dumps(advance(doc, polls[0], speed)).encode()
            polls[0] += 1
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # a screenshot run needs a quiet terminal
            pass

    return Handler


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8674)
    ap.add_argument("--file", default=str(ROOT / "engine/tests/data/client/status-decode.json"))
    ap.add_argument("--speed", type=float, default=96.0, help="tokens per second of the first running request")
    a = ap.parse_args(argv)
    if a.port == REAL_PORT:
        print("port 8662 is the real server's", file=sys.stderr)
        return 2
    doc = json.loads(Path(a.file).read_text(encoding="utf-8"))
    HTTPServer(("127.0.0.1", a.port), make_handler(doc, a.speed)).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
