# Does the 512-expert Flash-Next replace the Coder as the 64 GB recommendation? Agent quality and MTP acceptance (#146, phase 2b)

## Question and decision

Speed is settled (`2026-11-15-pc-flash-next-512-speed.md`: **decode** 0.89 to 1.35 of the Coder, **prefill** equal, MTP
tokens per pass 6.71 / 3.15 / 4.04 against 6.40 / 3.45 / 2.74). What is open: is the 512 file (ISTA GSQ-RCO Q2_0, all 512
experts, dense tensors repacked losslessly) a better or an equal coding-agent worker than the Coder (IQ1_M, 256 experts)?
Reference numbers of the Coder, same harness, same cases: BFCL multi-turn **25/52** with the parser fix
(`2026-11-15-pc-bfcl-tool-path.md`), SWE-bench Verified subset **12/17** (`2026-11-16-pc-dm-vs-coder.md`, before the parser
fix; mini-swe-agent uses text commands, not tool calls, so the fix does not reach it).

| Outcome (512 file against the Coder) | Decision |
|---|---|
| BFCL at least 5 cases above 25 (30/52 or more) or SWE at least 2 tasks above 12 (14/17 or more), and neither below its threshold | the 512 file becomes the 64 GB recommendation; the Coder stays as the smaller file (less RAM) |
| both within the thresholds (BFCL 21 to 29, SWE 11 to 13) | equal: the Coder stays the recommendation (smaller, 26 GB of experts pinned against 35 GB, already shipped); the 512 file is documented as the alternative |
| BFCL 5 or more below, or SWE 2 or more below | worse: the Coder stays; the 512 file is not offered |
| MTP acceptance (tokens per pass of the BFCL and SWE server logs) clearly below the Coder's | only then is a head trained for 512 experts a question; otherwise the Coder's head stays |

Thresholds are those of #145 (BFCL 5 cases, SWE 2 tasks); one pass of 17 tasks has a standard deviation of the difference of
about 2 tasks, so only large gaps decide.

The DM re-run (Coder cut to 164 of 256 experts, `coder-DM-164e`) uses the fixed tool-call parser (#147): DM scored **3/52**
in #145 with 324 string `"False"` parameters (the bug). Question: how much of that gap was the parser? Outcome: DM within
5 cases of the Coder's 25 (20 or more) -> DM is a candidate for the low-RAM classes again (decision by the ticket of #145);
still far below -> the model, not the parser.

## Plan

### Scope (one pass per model and case, paired with the existing Coder numbers; queue job per model and phase)

1. `bench/polyglot/model_job.sh flash512-tona bfcl`: BFCL multi-turn, the same 52 cases (`bench/bfcl/ids.json`), temperature
   0.001, ctx 131072, MTP (the Coder's head `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, as in the speed run), `RAM_GB=58`
   (`TONA_RAM_GB` and `MemoryMax` 58e9: the 512 file needs the 58 GB class; the quality does not depend on the class),
   port 8091.
2. `model_job.sh dm-tona bfcl`: the same, DM file. `RAM_GB=58` as in #145.
3. `model_job.sh flash512-tona swe`: the 17 tasks of `bench/swe/subset.txt` in 9 chunks of 2 (the last of 1), mini-swe-agent,
   step limit 100, ctx 262144, T 1.0, top_p 0.95, top_k 20, rootless Podman (images pulled at idle priority and removed
   after the run), `RAM_GB=58`. Grading with `bench/swe/evaluate.py`.
4. MTP acceptance: tokens per pass and draft acceptance from the server logs of the BFCL and SWE jobs of the 512 file,
   against the Coder's in the existing logs of #145/#147 (and the speed run above).
Build `work/build/f512` (branch head of #146, contains the parser fix). The Coder is not re-run: its BFCL and SWE numbers
exist with the same harness and cases.

Why not more: one pass per case is what the Coder's 25/52 and 12/17 are, so the comparison is like for like; a second pass
or other sampling would buy a smaller threshold than the harness noise. GPU time: BFCL about 1 h per model, SWE about
25 min per task for the Coder (the 512 file's rate is similar), so about 7 h; the chunks make it resumable.

### Expectation (sources: #145, `2026-11-15-pc-bfcl-tool-path.md`, speed run of #146)

| Number | Expectation | Source |
|---|---|---|
| 512 BFCL | 25 to 35 of 52 (same architecture and template as the Coder, 2-bit experts instead of 1-bit, all 512 experts; the Coder 25) | Coder 25/52, Ornith 27/52 |
| 512 SWE | 9 to 14 of 17 (Coder 12) | #145 |
| DM BFCL with the fixed parser | 8 to 18 of 52 (3 with the bug; the Coder gained 8 from the fix; DM's own gap is the model) | #145, #147 |
| 512 MTP tokens per pass | within 0.8 to 1.2 of the Coder's (6.71 / 3.15 / 4.04 against 6.40 / 3.45 / 2.74 in the speed run) | speed run |
| Tolerance | BFCL +-3 cases, SWE +-2 tasks (one pass, MTP non-deterministic) | #147, #145 |

### Active tooling

`TONA_DIAG` default, no profilers, production mode (no `TONA_DETERMINISTIC`), `TONA_RAM_GB=58`, no `TONA_PIN_GB` or
`TONA_VRAM_GB`, no expert profile (cold, as in the speed run). The machine is shared with other workers (jobs through the
queue, their CPU work beside mine); `tools/hw_report.py` and `tools/quiet_check.py` before and after every job are in the
job's files (not committed; sums only, no program names).

## Results (2026-11-15 to 11-16; one pass per case, production mode, `TONA_RAM_GB=58`; build `f512` at the head of #146 on main 51d57d6 (BFCL, SWE chunks 0 to 8) and 4b4589b (SWE chunks 10 to 16; the Coder path is bit-identical, `isa-diff`))

Machine and conditions: Ryzen 5 9600X, RX 9060 XT, 60.5 GiB RAM (`hw_report` and `quiet_check` of every job in `/var/tmp/claude/146/results/tc-*`, not
committed). The quiet check warned before and after the jobs, as in #145 and #147 (other workers' CPU work beside mine, VRAM held by others
about 1 GiB); not waited for, as the plan said. Spread: one pass per case, unknown.

| Benchmark | 512 file (Q2_0, 512 experts) | Coder (IQ1_M, 256 experts) | DM (164 of 256 experts) | Gap 512 to Coder | Threshold |
|---|---|---|---|---|---|
| BFCL multi-turn, 52 cases, T 0.001 (parser fix #147) | **17/52** (base 6, long_context 5, miss_func 2, miss_param 4 of 13) | **25/52** (#147) | **4/52** (base, long_context, miss_func, miss_param 1 of 13 each; 3/52 before the fix) | **-8** | 5 |
| SWE-bench Verified subset, 17 tasks, T 1.0 | **6/17** (4 empty patches, 7 applied but not resolved) | **12/17** (#145: 2 empty patches, 3 not resolved) | 9/17 (#145) | **-6** | 2 |
| MTP **tokens per pass**, BFCL server log (tokens weighted) | 5.15 | 5.27 to 5.33 | 7.67 (#145 7.7) | 0.97 to 0.98 | not clearly lower |
| MTP **tokens per pass**, SWE server logs | 5.28 (9 jobs, 5.01 to 5.44) | 4.6 to 5.3 per job (#145 logs, 6 of 9 read) | - | about equal | - |

Resolved SWE tasks of the 512 file: django-15268, django-16082, django-9296, django-14559, django-15563, xarray-4629. Empty patches (the agent ended
without a submission): django-10554, django-13516, sphinx-9461, django-11477. The Coder resolved 12 of the same 17 tasks.

### Reading against the plan

- **BFCL 17/52 against the plan's 25 to 35 (tolerance +-3): below the band**, 8 under the Coder, over the threshold of 5.
- **SWE 6/17 against the plan's 9 to 14: below the band**, 6 under the Coder, over the threshold of 2.
- **DM with the fixed parser 4/52 against the plan's 8 to 18: below the band.** The parser fix (#147) moved DM from 3 to 4 cases only: its low BFCL score is the model, not the
  parser (the Coder gained 8 from the same fix). DM stays under "far below" for the low-RAM classes, as #145 decided.
- **MTP acceptance** of the 512 file with the Coder's head is not lower (tokens per pass 0.97 to 0.98 of the Coder's in BFCL, about equal in SWE), so a head trained for 512 experts is not a question.
- Why the 512 file scores below the Coder was not examined (**ungeprueft**): possible are the 2-bit quantization of the full 512 experts against the 1-bit cut Coder, the dense
  tensors repacked to Q6_K/Q8_0 (lossless against the source), or the sampling at T 1.0. A look at the failed tasks and cases would be a separate question.

### Decision (table of this file)

BFCL 5 or more below and SWE 2 or more below: **worse**. **The Coder stays the 64 GB recommendation; the 512 file is not offered.** The speed
result stays as measured (**decode** 0.89 to 1.35 of the Coder, **prefill** equal), so speed gives no reason to switch. A different quantization of the 512-expert
model would be a new question with its own plan.
