# threshold: does the 0.01-nat quality rule predict task accuracy?

Dose-response benchmark for [research/31](../../research/31-nll-threshold-validation.md): graded degradation
(`TONA_IDX_SHARE`, decode only) against Δ-NLL and task success on the same tasks. The pilot measures the flip rate f (and
so the number of tasks): `research/measurements/2026-10-30-threshold-pilot.md`.

## Design

- **Two foreign repositories** at pinned commits (`common.py`), fetched by `fetch.py` to `work/tmp/w76/src`, never copied
  into this repository: pydash (MIT), boltons (BSD-3-Clause).
- **One 64K-token context per repo**, identical for all tasks (system message), so the prompt cache computes the prefill
  once. Whole files in a fixed order until the budget, the last one cut at a top-level statement. Token counts from
  `tona-tokenize`.
- **Family B (code, tests)**: 30 function bodies per repo are replaced by `...` in the shared context (`holes.py`: at
  least 80 % covered by the suite, 3 to 35 body lines, a test fails when the body is `raise NotImplementedError`, spread
  evenly). Task: write the function. Check: the reply's body replaces the original in a copy of the repo (other holes
  keep their bodies) and the **whole test suite** must pass in the sandbox.
- **Family A (retrieval, exact answer)**: generated from the AST of the holed context (`gen_a.py`): default value of a
  parameter, module-level constant, first base class, the one function that calls X, function by docstring start,
  return annotation. Exact match after normalisation (`answers.py`). Each source lies at least 2,400 tokens before the
  end (QSA selects nothing below 2,051 positions); about 10 % are local controls with the source in the last 2,000 tokens.
- Greedy, thinking off, one JSONL line per task (`run.py`).

## Selection rule (fixed before the first variant run)

By reference behaviour only (`select_tasks.py`): a task is kept when the **selection run** (a reference run) solved it
and, for family A only, the **closed-book run** (repository removed from the prompt) did not. The first drops tasks a
variant cannot lose, the second tasks solvable without the long context. The runs the variants are paired against are
another reference run (avoids regression to the mean). f0 is computed on all tasks.

## Statistics (`analyze.py`, `stats.py`)

Per run and family, paired against the reference: `b` tasks lost, `c` gained, f = (b + c) / n, exact McNemar test,
Newcombe score interval (method 10) for the paired accuracy difference, median seconds per task. Smallest detectable
difference at 80 % power, alpha 0.05: d = 2.8 sqrt(f / n) (research/31, section 3.4).

## Sandbox: foreign code runs only here

Tests of the foreign repositories run in a rootless Podman container: no network, read-only root, `/tmp` as tmpfs, the
repository mounted read-only (each run copies it to the tmpfs and applies the replacement there), all capabilities
dropped, no new privileges, memory, CPU, process and time limits. The image holds only the dependencies (pytest,
coverage, typing_extensions; `Containerfile`); the repositories are never installed. Outside the container the scripts
only read sources with `ast.parse`.

```sh
podman build -t localhost/tona-w76-py -f bench/threshold/Containerfile bench/threshold   # only step with network
podman run --rm -i --network=none --read-only --tmpfs=/tmp:rw,size=512m,exec --user=65534:65534 --cap-drop=ALL \
  --security-opt=no-new-privileges --memory=2g --memory-swap=2g --cpus=2 --pids-limit=256 --timeout=140 \
  --volume=work/tmp/w76/src/pydash:/repo:ro,z localhost/tona-w76-py python -I -c "$(cat bench/threshold/sandbox_inner.py)" < payload.json
```

`sandbox.py` (`podman_cmd`) builds this command; `payload.json` has `pythonpath`, `test_args`, `overrides` ({path: new
file text}), `timeout`, `coverage`, `src_dir`.

## Run

```sh
python3 -I bench/threshold/fetch.py                       # clone at the pinned commits
for r in pydash boltons; do python3 -I bench/threshold/holes.py $r; python3 -I bench/threshold/build.py $r
  python3 -I bench/threshold/gen_a.py $r 150; python3 -I bench/threshold/gen_b.py $r; done
python3 -I bench/threshold/score.py --selftest pydash     # the original bodies must pass (also: boltons)
# GPU: hold the lock for a whole variant run; TONA_* variables go to the server
flock "${TONA_GPU_LOCK:-work/gpu.lock}" bench/threshold/with_server.sh LOG 69632 \
  python3 -I bench/threshold/run.py http://127.0.0.1:8092 LABEL pydash,boltons
python3 -I bench/threshold/score.py LABEL                 # family B in the sandbox
python3 -I bench/threshold/analyze.py REF LABEL... [--select FILE]
python3 -m unittest discover -s bench/threshold           # pure parts, no model
```

Raw data is under `work/tmp/w76/` (override with `W76_WORK`).

Pilot (`pilot_all.sh`, one build, `BUILD=w76`): ref1, l0 (must differ from ref1, `check_differs.py`), closed (A only),
ref2 (`TONA_DRAFT_N=1`: the server has no mode without a drafter), d8.

## Main run (part 2)

- Tasks: 300 questions A per repo; family B = **every** qualifying function (116 in pydash, 176 in boltons), all holes of
  the one shared context (about 62.3K tokens, below 64K). Reply limit for B: 2,000 tokens (700 cut 37 % of the pilot
  replies).
- Runs (`main_run.sh`, one build): `ref1`, `ref2` (`TONA_DRAFT_N=1`, f0), then stages in `TONA_IDX_SHARE` with 8, 6, 5, 4,
  2, 1 full layers (12 is `ref1`); layer 0 is always full, the rest spread over the 12 QSA layers: 8 = 0,1,3,4,6,7,9,10
  (as in `research/measurements/2026-10-29-indexcache.md`), 6 = 0,2,4,6,8,10, 5 = 0,2,5,7,10, 4 = 0,3,6,9, 2 = 0,6, 1 = 0.
  Stages 6 and 4 once more with `--shuffle` (other task order, other cache history). Every variant must differ from
  `ref1` (`check_differs.py`).
- **Primary analysis: all tasks, unselected**, paired against `ref1` (`curve.py`): loss in pp with the 95 % Newcombe
  interval and exact McNemar test per stage and family. A stage is *damage* when the upper bound of the difference is
  below 0, *safe* when the lower bound is above the equivalence margin (A -3 pp, B -5 pp), else *open*. Six stages are six
  tests: one isolated hit is weak, the trend counts (Spearman).
- Secondary: the selection rule on the same runs (selection run `ref2`; B without closed-book). The pilot showed it
  leaves few B tasks; it is reported next to the primary.
- Quality axes per stage: delta-NLL (mean, 95 % block bootstrap), KL and top-1 from `TONA_TEACHER_TAIL=4096` runs on
  py / cpp / sh at 32K (`nll.sh`, `nll_summary.py`) and on the tail of both contexts. The curve is loss against pooled
  delta-NLL and against KL; the answer is the first *damage* stage and its delta-NLL relative to 0.01. If the curve only
  resolves "between x and y", report that.

## Comparing two shards of one model

`shard_run.sh` runs the same tasks on two model directories with one build and no variant switches (`W76_MODEL_DIR`,
`W76_PROFILE`, `W76_PORT` are read by `with_server.sh`): labels `old-ref1`, `new-ref1`, `old-ref2`, `old-closed`,
`new-ref2`. Analysis: `analyze.py old-ref1 new-ref1`. Plan and results: `research/measurements/2026-11-13-coder-1019-tasks.md`.
