# Cache admission in simulation: CRF admission and ClassLRU against LRU with prefetch, 2026-11-02 (#97)

## Plan (written before the first run)

Question: do the policies of llama.cpp #31286 (CRF admission) and #31233 (ClassLRU) beat our LRU with prefetch at the
back of the list on the route traces of 2026-10-25? Go into the server only with at least +1 point of union hit rate
on both traces (`research/36-llama-cpp-expert-streaming.md` section 6; about +1.5 % decode, an estimate).

- Scope: CPU only, no GPU, no server, no build. Two traces (`IE_ROUTE_TRACE_P`, two server runs of 2026-10-25, 8 coding
  requests, speculation on): `trp1.bin` (231,662,784 bytes, 610 passes) and `trp2.bin` (253,296,576 bytes, 682
  passes). 90 slots per layer, hits counted over the unions from 20 % of the run, preloading as in the server (top-16
  per token from the prediction, 8 candidates, at the back of the list, predicted ones protected). One simulation
  run is deterministic: no rounds, no spread; the spread is the difference between the two traces.
- Policies: (1) LRU with preloading, the reference; (2) LRFU with preloading (decayed hit count per pass, H = 3 / 10 /
  30), the earlier simulation; (3) ClassLRU as in #31233 (CRF with half-life in tokens, classes = floor(log2 CRF),
  one LRU list per class, victim = list head with the maximum age / 2^class, CRF kept across evictions, clock = tokens
  of the layer); (4) the same with the admission of #31286 (a demand miss goes to the CPU instead of the cache if its
  CRF is not higher than the decayed CRF of its victim); (5) admission on plain LRU with the same rule. The source of
  3 and 4 is `src/llama-moe-cache.cpp` of the two PRs at the head commits (read through the GitHub API; the diff text,
  not the PR description, defines the algorithm). Preloaded experts get no CRF credit and sit at the front of class 0
  with infinite age (evicted first), the equivalent of "at the back of the list".
- Tuning against overfitting: the parameters (classes 4 / 8, half-life 16 / 32 / 64 / 128 / 256 / 512 tokens, admission
  on / off) are chosen on one trace and the chosen setting is then read on the other; both directions (tune on 1,
  check on 2, and the reverse). The decision uses the held-out number.
- Why this size: the grid is 2 x 6 x 2 = 24 settings plus 5 references per trace, each a pure-Python pass over about
  30,000 layer-passes (seconds to a minute); the traces are the only ones that exist and the GPU stays free.
- Expectation (tolerance): reproduction of "LRU with preloading" 91.9 / 90.1 % within +-0.3 point (tie order in
  `argsort` may differ; the old script is lost with the scratchpad, the tool is a port); LRFU 92.3-92.4 / 90.8-91.1
  (H = 3 / 10) as in the table of 2026-10-25-cache-policy.md, within +-0.3. ClassLRU and CRF admission: -0.5 to +1.0
  point over LRU with preloading (source: LRFU gained +0.4 to +1.0 in the same simulation; research/36 section 4.1
  expects 0 to +1.5 % decode, "most likely within noise"). A result above +1.0 on the held-out trace is a finding.
  Reproduction outside the tolerance: stop and report.
- Active tooling: `tools/cache_sim/` (this change), Python 3 standard library only, `nice -n 19`, traces copied to
  `/var/tmp/claude/wsim/`. No `IE_*` switches (no engine runs).

## Result

Tool: `tools/cache_sim/` (`python3 tools/cache_sim/cli.py --slots 90 --grid trp1.bin trp2.bin`, tests in
`tools/tests/test_cache_sim.py`). The old scripts (`softsim.py`) were in a scratchpad; the port reads the same files.
Raw output: the two tables below. Runtime: 9 s for the references, 18 s for the whole grid (4 to 6 processes).

### Reproduction (against the plan: within +-0.3 point)

| Hits over the unions, 90 slots | trp1 (run 1) | trp2 (run 2) | 2026-10-25 table |
|---|---:|---:|---|
| LRU with preloading | 91.87 | 90.12 | 91.9 / 90.1 |
| LRFU H = 3 / 10 / 30 | 92.31 / 92.38 / 91.41 | 90.75 / 91.08 / 90.44 | 92.3 / 92.4 / 91.4 and 90.8 / 91.1 / 90.4 |

Reproduced to the printed digit. The reference is 91.87 / 90.12.

### Policies, 90 slots, hit rate in % (difference to LRU with preloading in points)

| Policy | trp1 | trp2 |
|---|---:|---:|
| LRU with preloading (reference) | 91.87 | 90.12 |
| LRFU H = 10 passes (best of the earlier simulation) | 92.38 (+0.51) | 91.08 (+0.96) |
| LRU + CRF admission, half-life 16 / 64 / 512 tokens | 91.82 / 91.78 / 91.57 (-0.05 / -0.09 / -0.30) | 90.12 / 90.21 / 89.98 (0.00 / +0.09 / -0.14) |
| ClassLRU 4 classes, half-life 16 / 32 / 64 / **128** / 256 / 512 | 92.19 / 92.41 / 92.57 / **92.63** / 92.52 / 92.34 | 90.75 / 90.99 / 91.16 / **91.24** / 90.86 / 90.53 |
| ClassLRU 8 classes, half-life 128 / 256 | 92.56 / 92.49 | 91.23 / 91.18 |
| ClassLRU 4 classes + CRF admission, best (half-life 64) | 92.36 (+0.49) | 91.06 (+0.94) |
| ClassLRU 8 classes + CRF admission, best (half-life 64) | 92.33 | 91.05 |

Full grid (34 settings per trace): rerun the command above. Tuning crosswise (no overfitting):

- tuned on trp1: `class:4:128` (92.63, +0.76), held out on trp2: 91.24 (**+1.12**).
- tuned on trp2: `class:4:128` (91.24, +1.12), held out on trp1: 92.63 (**+0.76**).
- admission on LRU: tuned on trp1 `lru-admit:16` -> trp2 90.12 (+0.00); tuned on trp2 `lru-admit:64` -> trp1 91.78
  (-0.09).
- Both traces pick the same setting (4 classes, half-life 128 tokens = the default of #31233). 8 classes is level
  with 4 (<= 0.07 point).

### Slots (not in the plan, three more points of the same two traces; cost 10 s each)

| Slots | LRU | LRFU H = 10 | ClassLRU 4 / 128 | ClassLRU + admission 4 / 64 |
|---|---:|---:|---:|---:|
| 70 (desktop with 2.8 GB VRAM in use, 2026-10-25 note) | 86.35 / 85.11 | 87.94 / 86.60 | 88.15 / 86.66 | 87.94 / 86.60 |
| 90 (reference) | 91.87 / 90.12 | 92.38 / 91.08 | 92.63 / 91.24 | 92.36 / 91.06 |
| 110 | 94.88 / 93.49 | 95.02 / 94.11 | 95.18 / 94.18 | 94.94 / 94.05 |

Pairs are trp1 / trp2. The gain of ClassLRU over LRU shrinks with more slots (70: +1.80 / +1.55, 90: +0.76 / +1.12,
110: +0.30 / +0.69). Over LRFU it stays at +0.0 to +0.25.

### Against the plan

- Expectation "-0.5 to +1.0 point": ClassLRU +0.76 / +1.12 (the second is 0.12 above the range, a finding, but still
  only the second trace); CRF admission -0.3 to +0.09, inside.
- Decision rule (>= +1 point on both traces): **not met**. The tuned ClassLRU gives +0.76 on trp1 (below) and +1.12 on
  trp2; the mean is +0.94. The admission rule of #31286 gives nothing, as expected from section 4.1 of
  research/36: our preloading at the back of the list already filters the cold misses, and the cold-miss bypass only
  costs hits (the lost ones are those that come back within a few passes).
- Compared with what we already tried: LRFU gave +0.5 / +1.0 in the same simulation and **no gain in the server**
  (2026-10-24-misses). ClassLRU is +0.25 / +0.16 above LRFU H = 10, inside the difference between the two traces.
  Two server runs with LRFU measured 0 +- 0.5 % decode; +0.9 points of hits is about +1.4 % by the estimate of
  research/36 (0.65 ms per point of 42 ms), which a server A/B (decode sd 0.2-0.6 %) could just resolve, but the
  simulation ignores what the server pays for the policy: per-layer bookkeeping on the host, and the CPU-computed
  misses that refill nothing (`IE_FILL=0`), which change which experts are "used".
- Limits of the simulation: unions per layer and pass only (no timing, no CPU/GPU split of the misses, no refill
  rule, fixed 90 slots without the ramp-up), two traces from 8 coding requests each at <= 600 tokens (short
  context), tie order of equal logits may differ from the server; the policies are ports of the PR source at its head
  commit (the CRF admission of #31286 exists there only for its auto CPU share mode, where a bypassed miss is
  computed on the CPU; in our planner the same miss goes to the CPU or is read over PCIe, so the port is the
  nearest equivalent, not the same mechanism).

### Recommendation

No server work now. At the 90 slots of the reference the best policy (ClassLRU 4 classes, half-life 128 tokens) is
+0.76 / +1.12 points over our LRU, below the bar on one trace and not above LRFU, which did not show up in the server.
The one place where it would pay is fewer slots (70: +1.5 to +1.8 points, same as LRFU), i.e. machines or sessions with
less free VRAM, and where misses rise with context (below). If it is built after all, take LRFU/ClassLRU as one
change, 4 classes, half-life 128 tokens, no admission, and judge by `ab.py` `short` + `set` at 70 and 90 slots.

## Second question: route trace at 1M (plan only, nothing run)

Question: do the misses of 13.4 % (512K) and 19.1 % (1M) (`2026-11-02-scale-run.md`) come from routing that changes
with context (more distinct experts per pass, less reuse), so that a policy would help more than at 600 tokens?

- Smallest run that answers it: one trace of the decode at 1M and one at 512K, in the scale run that starts anyway
  (workload `huge`, `IE_SEQS=1`, `--ctx 1048576`), as an extra start after the A/B rounds, not inside them (the trace
  writes 2 x 1 KB per token and layer and changes timing: never a rate arm). Per start: prefill 17 min (512K: 6 min)
  that the scale run pays anyway when the prompt cache restore of that prompt is not used; decode 1,500 tokens, about
  550 passes at 2.7 tokens per pass and 50 ms: 30 s. Trace size: about 550 passes x 48 layers x (2 + 2 x 1024 x 2.7 B) =
  about 145 MB, written to `/var/tmp/claude/wsim/` or `work/`, not `/tmp` (rule: RAM).
- Do not trace the prefill: 1M tokens x 48 layers x 2 KB = about 98 GB. The record's `nb` is one byte, so prefill
  groups cannot be in this format anyway; before the run read whether the prefill passes reach
  `ExpertSelect::record_traces` (`engine/gpu/mega_select.cpp`, called from `mega_serve.cpp:191`); if they do, gate the
  write on `nb_ <= 16` (a four-line host change, C1-like, no number changes; the build is the only cost).
- Size of the decode (why 1,500 tokens): the 610 / 682 passes of the existing traces gave the spread of 1.8 points
  between traces (91.87 against 90.12) and the policy differences are 0.5 to 1.2; fewer than 500 passes would not
  separate them. 20 % are the warm-up (as in the simulation).
- Expectation (tolerance): LRU with preloading at 90 slots (the 1M start has 70 slots: also simulate at 70): hits
  **74-80 % over the unions** at 1M, from 19.1 % misses of expert accesses in the scale run (unit as logged there; the
  union rate is lower than the per-token rate because the union deduplicates) and the 11-14 points between
  600 tokens and 1M seen in 2026-10-24-misses (rust 76 % at 13K prompt). ClassLRU: +0.5 to +2.5 points over LRU at 70
  slots; a gain of at least +1.5 points at 1M is what would justify the ticket "ClassLRU/LRFU in the server".
- Active tooling for that start: `IE_ROUTE_TRACE_P=<file>`, `IE_DIAG=1`, `IE_SEQS=1`, as the scale run; the start is
  never used for a rate. Evaluation: `tools/cache_sim/cli.py --slots 70 --grid <trace>` plus the crosswise reading
  against trp1 and trp2.
- No GPU time is used by this plan; the GPU time of the start is requested in the plan of the next scale run
  (`research/measurements/*-scale-run*.md`).
