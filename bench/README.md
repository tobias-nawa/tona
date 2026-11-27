# bench: measurement harness (ibench)

Measures engines the same way through their HTTP API: Tona, llama.cpp, any OpenAI- or Anthropic-compatible server.
Python standard library only.

```sh
bench/run run --target tona                         # whole suite bench/suites/base.toml
bench/run run -t tona -c short-code,needle-10k -n 3  # only two cases, 3 runs each
bench/run compare research/measurements/A.json research/measurements/B.json
bench/run proxy --upstream http://127.0.0.1:8662 -o bench/traces/session.jsonl
bench/run env -t tona                               # only print the environment
```

Result per run: `research/measurements/<date>-ibench-<target>-<suite>.md` plus a `.json` with the raw data (`-2`, `-3` ...
is appended if the name exists).

Tests (no GPU): `cd bench && python3 -m unittest discover -s tests -t .`

## Levels

1. **Map**: micro-benchmarks in HIP (`bench/micro/`), not part of this harness.
2. **Engine**: this harness. Fixed cases, several runs, median and spread.
3. **Real work**: an agent solves tasks; the proxy records the session and the harness replays it as a trace.

## Metrics

Every request is streamed. The clock is `time.perf_counter()` in the client.

| Metric | Definition |
|---|---|
| `ttft_s` | Send until the first event with output (answer, reasoning or tool call) |
| `prefill_tps_client` | (prompt tokens - cached tokens) / TTFT. Includes HTTP and the first token |
| `prefill_tps_server` | `timings.prompt_per_second` of the server (llama.cpp field names; other servers use the same) |
| `decode_tps_client` | (output tokens - 1) / (last - first chunk). Only from `min_decode_tokens` on |
| `decode_tps_server` | `timings.predicted_per_second` |
| `tokens_per_pass` | `predicted_n / (predicted_n - draft_n_accepted)`: tokens per verify pass with MTP |
| `draft_accept_rate` | `draft_n_accepted / draft_n` |
| `itl_p50_ms`, `itl_p99_ms` | Gap between chunks. With MTP one chunk holds several tokens |
| `vram_peak_mib`, `ram_avail_min_mib`, ... | from sysfs and `/proc/meminfo`, every 100 ms, over the duration of the request |

Client and server values are both reported; a strong difference is a finding.

## Cache control

- `cache = "cold"`: a nonce in the system prompt prevents prefix hits; the report warns if the server reports cached tokens.
- `cache = "warm"`: same prefix in every run, only the end of the question changes; the warm-up run fills the cache.
  Like an agent session; mostly TTFT is measured.

## Writing cases

A suite is a TOML file under `bench/suites/`. Unknown fields are an error. Every case has exactly one source:

```toml
[[case]]
name = "needle-10k"
context_chars = 34000          # excerpt of bench/corpus/, cut at a line boundary
needle = "kestrel-4711"        # inserted as SECRET_DEPLOY_TOKEN
needle_depth = 0.5             # position in the context, 0 = start
question = "..."
expect = 'kestrel-4711'        # regex on the answer: measures quality on the side
max_tokens = 768
```

Instead of `context_chars`, use `prompt = "..."` or `trace = "../traces/x.jsonl"`.
More fields: `runs`, `warmup`, `cache`, `temperature`, `min_decode_tokens`, `pause_s`, `extra` (additional body fields),
`description`; all also work under `[defaults]`. The context size is in characters so a case is the same for every
tokenizer; the report has the real token count.

## Targets

`bench/targets.toml`: `url` (without `/v1`), `api` (`openai` or `anthropic`), `model`, `extra`. `extra` means something
different on every server (e.g. reasoning); both sides must reason equally much, check `reasoning_tokens` or
`reasoning_chars` in the JSON.

Local targets go into `work/bench/targets.local.toml` (git-ignored, same format, wins on a name clash).

## Traces

`bench/run proxy` passes everything through and writes every request to `/v1/chat/completions` or `/v1/messages` as a
JSON line. Example with Claude Code:

```sh
bench/run proxy --listen 127.0.0.1:8099 --upstream http://127.0.0.1:8662 -o bench/traces/x.jsonl
ANTHROPIC_BASE_URL=http://127.0.0.1:8099 claude
```

Traces are git-ignored (they contain session code). On replay all steps of a run share one nonce: prefix hits between
steps, none between runs.

## Corpus

`bench/corpus/python-stdlib-3.14.txt`: five CPython 3.14 standard library files, 405 KB, licence in
`bench/corpus/LICENSE-python.txt`. Do not change it; old measurements would no longer compare. Changes against
CPython (PSF licence, clause 3): `Lib/argparse.py`, `Lib/subprocess.py`, `Lib/logging/__init__.py`,
`Lib/asyncio/base_events.py` and `Lib/asyncio/tasks.py` joined into one file, each after a line
`# ==== file: <path> ====`; their copyright lines stay.
