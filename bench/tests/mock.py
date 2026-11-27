"""Mock server with a precisely timed SSE stream, for testing the measurement logic."""

import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class Mock:
    """first_s: time to the first token; n: tokens; gap_s: gap after that; per_chunk: tokens per chunk (MTP)."""

    first_s: float = 0.2
    n: int = 11
    gap_s: float = 0.02
    per_chunk: int = 1
    text: str = "def fib(n):\n    pass\n"
    prompt_tokens: int = 1000
    cached_tokens: int = 0
    timings: bool = True

    def __post_init__(self):
        self.requests: list[dict] = []
        mock = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps({"data": [{"id": "mock"}]}).encode() if self.path == "/v1/models" else b"{}"
                self.send_response(200 if self.path == "/v1/models" else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                mock.requests.append(req)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                if self.path == "/v1/messages":
                    mock._anthropic(self)
                else:
                    mock._openai(self)
                self.wfile.write(b"0\r\n\r\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    @staticmethod
    def _send(h, data: str, event: str | None = None):
        raw = (f"event: {event}\n" if event else "") + f"data: {data}\n\n"
        b = raw.encode()
        h.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
        h.wfile.flush()

    def _pieces(self):
        """(pause before, text) per chunk: the n tokens as n/per_chunk chunks."""
        chunks = max(1, self.n // self.per_chunk)
        step = max(1, len(self.text) // chunks)
        for i in range(chunks):
            piece = self.text[i * step : (i + 1) * step] if i < chunks - 1 else self.text[i * step :]
            yield (self.first_s if i == 0 else self.gap_s), piece or "x"

    def _openai(self, h):
        for pause, piece in self._pieces():
            time.sleep(pause)
            self._send(h, json.dumps({"choices": [{"index": 0, "delta": {"content": piece}}]}))
        self._send(h, json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
        last = {
            "choices": [],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.n,
                "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
            },
        }
        if self.timings:
            last["timings"] = {
                "prompt_per_second": 800.0,
                "predicted_per_second": 50.0,
                "predicted_n": self.n,
                "draft_n": 9,
                "draft_n_accepted": 6,
                "cache_n": self.cached_tokens,
            }
        self._send(h, json.dumps(last))
        self._send(h, "[DONE]")

    def _anthropic(self, h):
        self._send(
            h,
            json.dumps(
                {
                    "type": "message_start",
                    "message": {
                        "usage": {
                            "input_tokens": self.prompt_tokens - self.cached_tokens,
                            "cache_read_input_tokens": self.cached_tokens,
                            "output_tokens": 1,
                        }
                    },
                }
            ),
            "message_start",
        )
        self._send(
            h,
            json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            "content_block_start",
        )
        for pause, piece in self._pieces():
            time.sleep(pause)
            self._send(
                h,
                json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}}),
                "content_block_delta",
            )
        self._send(
            h,
            json.dumps(
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": self.n}}
            ),
            "message_delta",
        )
        self._send(h, json.dumps({"type": "message_stop"}), "message_stop")
