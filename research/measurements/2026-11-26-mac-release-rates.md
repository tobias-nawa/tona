# Mac: decode and prefill of the release start on the M1 Max (2026-11-26)

## Plan (written before the run)

- **Question**: which decode rate (and prefill rate) has exactly the server that the release installs on the M1 Max
  32 GB, at the contexts of the Coder's rates in `data/modeldb/offer.json` (2K, 16K, 64K, code)? The installer shows no
  measured number for this model on the Mac (`installer/modules/metal_apple.cpp` has one only for `coder-dmt-164`).
  A narrow range becomes a proposal for the installer; a wide one means: still no number.
- **Server** (the environment `start-server.sh` gets from `tona-install` of `feat/macos-release` 723b7b1e on this Mac,
  read from its `--dry-run`: model `coder-iq1_m`, drafter `mtp`, room "Tona up to 19 of 32 GB", priority balanced):
  build `work/build/rates` of 723b7b1e (Release, Metal, deployment target macOS 15, as `release-build-macos.sh`);
  `coder-iq1_m` part 1 with `--mtp` `mtp-q8_0` (the pinned head, all 512 experts, 4.1 GB), `--ctx 32768`;
  `OMP_NUM_THREADS=6 TONA_SEQS=1 TONA_RAM_GB=19 TONA_VRAM_GB=26.8 TONA_OEA_K0=0`, expert profile
  `toolchain/profiles/coder-iq1_m.bin` (the file the installer embeds), `TONA_PRIORITY` unset (= balanced).
  - **Deviation**: no `TONA_PROMPT_CACHE`. The release writes prompt checkpoints to the SSD; across rounds the same
    prompt would then be read back instead of prefilled. Without it every request pays its full prefill, as a new
    prompt does in the release. The cost of writing a checkpoint is not in these numbers.
  - **64K is not measured**: the release starts with `--ctx 32768`, a 64K prompt does not fit. No number at 64K.
- **Run**: `scripts/ab.py` of 723b7b1e, A = B = this build and environment (an A/A: the two arms give the spread),
  workload `server`, targets `2000:200,16000:100` (the contexts and token counts of
  `2026-11-20-coder-decode.md`, the source of the offer rates), code corpus `work/corpus/code.txt`, 3 rounds fixed
  (`--fixed-rounds`; no Mac entry for these metrics in `bench/noise.toml`, so at least 2; 3 give 6 starts per
  target), warm-up start and a discarded start of each arm per round (ab.py's default on macOS), load log on.
  Duration: about 40 min (per start: load about 30 s, 2K about 25 s, 16K about 2.5 min; ab.py prints its estimate).
- **Expectation** (sources: `2026-11-25-mac-priority-check.md` arm D-bal, budget 10.77 GB, decode 21.5 tok/s
  (20.2 to 21.6) on 200-token code requests; `2026-11-17-mac-iq1m-reserve.md` budget 8.8 GB, 16.7 tok/s; the release
  check of 2026-11-26 started this setup with budget 10.88 GB and prefilled a 13,000-token prompt at 126 tok/s cold):
  - decode@2000: 18 to 25 tok/s (the MTP head with all 512 experts may accept more than the 256 cut of the earlier
    runs; the budget is the same within 0.1 GB);
  - decode@16000: 15 to 23 tok/s (attention over 16K costs more per pass on the Mac than on the PC);
  - prefill@2000: 100 to 300 tok/s, prefill@16000: 90 to 200 tok/s (wide: no prefill of this setup at these sizes);
  - A against B: within plus or minus 5 % for every rate (same build and environment).
- **Decision rule**: a rate becomes a proposal for the installer only if all 6 starts lie within plus or minus 10 %
  of their median (the installer shows a measured number with plus or minus 10 %); otherwise that rate gets no number.
- **Active tooling**: ab.py's load log (one sampler beside each start), round warm-up; no `TONA_DIAG`, no
  `TONA_DETERMINISTIC`, no profiler, no other `TONA_*` variable than those above.
- **Machine**: quiet check before and after; on a warning, the machine is quieted once and the run starts anyway,
  with the load reported as sums.

## Result (2026-11-26, 01:30 to 02:02)

Quiet check before (01:29:11, again at 01:29:44 after a backup was stopped): **do not measure**, other programs used
20.1 and 28.3 % of all CPU threads (load average 45.7 and 30.9; background work of the system that could not be
stopped). The run started anyway (maintainer rule: quiet once, then measure). After (02:02:51): **warning**, other
programs 6.3 % (load average 1.0). Load as sums from ab.py's load log, per start: other programs 60 to 66 % of one
core in starts 1 to 4, 25 % in start 5, 9 to 10.5 % from start 6 on; the system 13 to 113 %; our server 16 to 19 % of
one core and 73 to 80 % of the GPU's time; 0 to 1.8 MiB swapped in per start.

ab.py: `work/ab/2026-11-26-mac-release-rates.{md,json}`, server logs `work/ab/2026-11-26-mac-release-rates-logs/`.
3 rounds (AB, BA, AB), 32.6 min, 13 server starts (1 warm-up, 6 discarded, 6 measured). Every output SHA equal in
both arms and all rounds (2K: `34926e2aa8162172`, 16K: `4e89a7a1806666c9`); tokens per pass 3.08 at 2K, 2.70 at 16K
in every start. A against B: prefill@2000 -0.27 %, decode@2000 +0.74 %, prefill@16000 -2.26 %, decode@16000 -2.74 %,
all inside the plan's plus or minus 5 %.

| Rate | The 6 measured starts | Median | Range | Largest deviation from the median | Plan |
|---|---|---|---|---|---|
| decode@2000 | 17.4, 17.9, 17.9, 17.7, 17.9, 17.9 | **17.9 tok/s** | 17.4 to 17.9 | 2.9 % | 18 to 25: just below |
| decode@16000 | 18.0, 17.9, 18.0, 18.0, 17.9, 16.6 | **17.9 tok/s** | 16.6 to 18.0 | 7.4 % | 15 to 23 |
| prefill@2000 | 156.2, 157.0, 158.3, 156.6, 160.6, 160.3 | **157.6 tok/s** | 156.2 to 160.6 | 1.9 % | 100 to 300 |
| prefill@16000 | 134.3, 125.7, 136.6, 135.8, 136.6, 137.0 | **136.2 tok/s** | 125.7 to 137.0 | 7.7 % | 90 to 200 |

(The 2K prompt is 1,934 tokens, the 16K prompt 15,575, as the server counts them.)

Read against the plan:

- **By the decision rule every rate is narrow** (all 6 starts within plus or minus 10 % of the median): a proposal for
  the installer, measured on the M1 Max 32 GB with the release start: decode 17.9 tok/s at 2K and at 16K, prefill
  158 tok/s at 2K and 136 tok/s at 16K. **64K: no number** (does not fit `--ctx 32768`).
- **The arena budget of every measured start was the RAM limit's**: 13.53 GB (`TONA_RAM_GB=19` less the reserve of
  3.0 GB and what the process holds), at start 13.8 to 15.1 GB free. So these numbers are what the release reaches when
  the limit binds; a machine where other programs hold more than about 13 GB at the server's start gives a smaller
  budget.
- **Outside the expectation (a finding): decode@2000 at 17.9 lies just below 18 to 25**, although the budget (13.5 GB)
  was larger than in the runs the expectation came from (10.77 GB, 21.5 tok/s, MTP head cut to 256 experts). The
  difference to those runs is the head: the release ships the pinned `mtp-q8_0` with all 512 experts (4.1 GB), the
  earlier runs used the 256 cut (2.8 GB). Unchecked: whether the larger head is the cause (a pass of the full head reads
  more, and its 1.3 GB more of GPU buffers leave less of the working set to the experts).
- **The context hardly changes decode** (17.9 at 2K and at 16K; 2.70 against 3.08 tokens per pass is offset by
  cheaper passes, unchecked why).
- **The first start after the builds was much slower** (warm-up, discarded): budget 10.85 GB with a cold page cache,
  decode 10.6 tok/s at 2K and 13.1 at 16K. A user's first requests after a restart of the Mac may look like that; the
  numbers above are those of a warm machine.
- **Load**: starts 3 and 5 (other programs 60 to 66 % and 25 % of one core) decoded as fast as the quiet later starts
  (17.4 to 17.9 tok/s), so the load before the run did not move the measured rates by more than their spread.

Test conditions before:

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 45.68, 23.46, 14.92
- CPU use by other programs over 0.5 s: 27.3 % of all threads
- sandbox: no
- Commit: `723b7b1e` (clean tree)
```

Test conditions after:

```
### Test conditions

- CPU: Apple M1 Max, 10 cores / 10 threads
- GPU: Apple M1 Max, 32.0 GiB unified memory
- RAM: 32.0 GiB
- OS: macOS 27.0.1, kernel 27.0.0 (arm64)
- Driver and toolkit: no toolkit found
- TONA_* variables: none set
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 1.03, 1.06, 2.66
- CPU use by other programs over 0.5 s: 5.8 % of all threads
- sandbox: no
- Commit: `723b7b1e` (clean tree)
```
