"""Recording proxy: sits in front of a server, passes requests through and writes completion requests as trace lines."""

import contextlib
import http.client
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

RECORD = {"/v1/chat/completions": "openai", "/v1/messages": "anthropic"}
# Headers that are not forwarded; http.client sets the length, or the connection end does
HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def make_handler(upstream: str, out_path: str):
    up = urlsplit(upstream)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0: end of response = end of connection, so streamed responses without a length pass through
        protocol_version = "HTTP/1.0"

        def log_message(self, fmt, *args):
            print(f"[proxy] {self.command} {self.path} {fmt % args}", file=sys.stderr)

        def do_GET(self):
            self._forward()

        def do_POST(self):
            self._forward()

        def _record(self, body, api):
            try:
                obj = json.loads(body)
            except ValueError:
                return
            line = json.dumps({"t": time.time(), "api": api, "path": self.path, "body": obj}, ensure_ascii=False)
            with lock, open(out_path, "a") as f:
                f.write(line + "\n")

        def _relay(self, resp):
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.end_headers()
            while chunk := resp.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()

        def _forward(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else None
            api = RECORD.get(self.path.split("?")[0])
            if api and body:
                self._record(body, api)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            conn = http.client.HTTPConnection(up.hostname, up.port or 80, timeout=900)
            try:
                conn.request(self.command, up.path.rstrip("/") + self.path, body, headers)
                self._relay(conn.getresponse())
            except (OSError, http.client.HTTPException) as e:
                with contextlib.suppress(OSError):
                    self.send_error(502, f"Upstream: {e}")
            finally:
                conn.close()

    return Handler


def serve(listen: str, upstream: str, out_path: str) -> ThreadingHTTPServer:
    host, _, port = listen.rpartition(":")
    return ThreadingHTTPServer((host or "127.0.0.1", int(port)), make_handler(upstream, out_path))
