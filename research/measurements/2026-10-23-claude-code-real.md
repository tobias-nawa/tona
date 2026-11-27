# Claude Code in real use with ie-server (2026-10-23, 21:14–21:17, log from an interactive terminal)

`just serve` (context 262,144, KV int8, DFlash drafter, PCIe x16), Claude Code in a separate test project (.NET/Blazor planning).

| Prompt | recomputed | Prefill | Output | Decode | Tokens/pass |
|---:|---:|---|---:|---:|---:|
| 20,774 | 20,774 | 38.3 s (542 tok/s prefill) | 136 | 28.8 tok/s | 2.54 |
| 21,324 | 413 | 1.1 s | 229 | 37.0 | 3.16 |
| 21,613 | 58 | 0.5 s | 175 | 43.1 | 3.69 |
| 21,848 | 58 | 0.4 s | 186 | 35.5 | 3.07 |
| 23,777 | 1,742 | 3.4 s | 407 | 30.6 | 2.55 |
| 25,446 | 1,259 | 2.6 s | 402 | 35.9 | 3.09 |
| 25,910 | 59 | 0.5 s | 165 | 34.6 | 3.02 |
| 26,132 | 56 | 0.5 s | 1,216 | 39.7 | 3.41 |
| 27,405 | 54 | 0.4 s | 255 | 35.1 | 3.11 |

- The prefix cache takes effect from the second request: only the new tokens are computed.
- Decode at 21–27K context 29–43 tok/s.
- Prefill of the first prompt, 20,774 tokens long, only 542 tok/s against 969 at 6,000 tokens, cause not yet measured. Presumably: the late chunks
  compute attention over more cells, and the first batches run with a cold expert cache. Next lever for prefill.

## Re-measurement 22:10–22:20 (page cache warm)

| Run | Tokens | tok/s (prefill) |
|---|---:|---:|
| ie-gpu-gen, C++ source, context 32K | 6,000 / 12,000 / 20,000 / 20,000 | 887 / 791 / 823 / 926 |
| ie-gpu-gen, context 262K | 20,000 ×2 | 923 / 924 |
| ie-gpu-gen, context 262K + DFlash (85 slots like the server) | 20,000 | 914 |
| ie-gpu-gen, text of the Claude Code request (English + tool JSON) | 20,000 | 854 |
| ie-server like `just serve`, one message | 19,383 | **919** (21.1 s) |
| ie-server, recorded Claude Code request (hang-repro) | 7,457 / 7,406 | 867 / 975 |

- Neither context size, drafter nor kind of text explains the 542 tok/s. Most likely cause: cold page cache
  after the restart at 21:05 (unpinned experts from the NVMe). The first run after the restart also reached
  only 582 tok/s. Not measured directly.
