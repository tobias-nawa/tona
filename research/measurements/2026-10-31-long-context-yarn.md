# Long context: YaRN switch, cost below 262K, retrieval at 512K and 1M (#77, phase 1)

Date: 2026-10-30 and 2026-10-31. Build `work/build/w77` (branch `77-long-ctx`, main c1b42e3 merged), reference build
`work/build/merged` (main c1b42e3) and `work/build/main` (c25f9b2, only for the first bit-exact check). Model Coder IQ1_M,
MTP head, desktop on the GPU. Raw data (outputs, corpora, scripts) is in `/var/tmp/claude/w77/` (`out/`, `corp/`) and
`/var/tmp/claude/w76/runs/y-*.jsonl`; the per-token files are small, the tables below are derived from them.
Rates are named: **prefill** (prompt tokens per s) or **decode** (generated tokens per s, MTP).

## 1. The switch and what it costs when off

`IE_ROPE_YARN=off|<factor>`, `IE_YARN_MSCALE=<value>` (`engine/backend/rope_yarn.h`, `docs/use/environment.md`). Formula as
transformers `_compute_yarn_parameters` and llama.cpp `rope_yarn` (beta_fast 32, beta_slow 1, original 262,144, 64 rotated
dimensions, theta 1e7): pairs 0-14 unchanged, 15-21 blended, 22-31 divided by the factor; mscale 0.1 ln(f) + 1 (1.1386 at
f = 4). It applies to attention q/k, indexer q, the pooled indexer key, the MTP layer, prefill and decode, GPU and CPU
reference. The DFlash drafter stays native (no rope-scaling key in any of the three GGUFs).

- **Off is bit-identical.** `IE_DETERMINISTIC=1`, greedy, 200 tokens, server, same request sequence (30K prompt, then
  130K prompt): main c25f9b2, w77 before the merge, merged c1b42e3 and w77 after the merge give SHA `065922c2ff602991`
  (30K) and `439aad472c59ae6f` (130K) in every run (4 pairs). A single 130K request without the 30K before it gives
  `a90744c565420c39` on merged and on w77 (2 runs each): the request sequence changes the numbers (prefix reuse), so only
  equal sequences compare.
- **Golden against w77:** decode (csv, rust, explain), prefill with dflash (long) identical; golden-server (all 20 cases, exit 0)
  identical.
- **ISA:** only the rotation kernels differ (`k_qsa_prep`, `op_qsa_prep`, `k_mega<1/4/8>`, `k_idx_pool`,
  `k_head_norm_rope`); `k_mega<1>` has 2 more VGPR spills (56 to 58 at 192 VGPRs).
- **Speed with off, 3 rounds alternating merged / w77** (server, MTP, `--ctx 262144`):

| | merged | w77 (off) |
|---|---|---|
| bench-set code, **decode** tok/s | 87.9 / 88.9 / 88.9 | 89.2 / 86.9 / 88.8 |
| bench-set text, **decode** tok/s | 64.0 / 64.7 / 64.7 | 64.7 / 63.0 / 64.7 |
| **decode** after a 128K prompt, tok/s | 63.1 / 63.9 / 63.8 | 63.9 / 62.3 / 63.8 |
| 128K **prefill**, tok/s | 1717 / 1718 / 1718 | 1718 / 1717 / 1718 |

  No difference outside the spread, so the off path was not built without cost.
- `--check` (CPU against GPU, "The capital of France is", 6 steps): off worst relative deviation 9.44e-02, YaRN x4
  7.89e-02, top-1 6 of 6 in both. At 6 tokens only mscale acts; the long-position agreement is covered by the
  teacher-forced runs below (same prefill and decode paths, YaRN on).

## 2. Step 2: 1M starts

`IE_SEQS=1 --ctx 1048576 IE_ROPE_YARN=4 IE_VRAM_LOG=1`, `systemd-run --user --scope -p MemoryMax=54G -p
MemorySwapMax=0` (60 GB machine minus 6 GB as in the other measurements), 53 GB available at the start:
loaded in 21.7 s; KV int8 14.50 GB in pinned RAM, VRAM 14.22 GB, **70** expert slots per layer (about 80 at 262K), 25.1 GB
pinned experts; server RSS 45.1 GB.

**Open (fixed in phase 2, research/measurements/2026-10-31-long-context-topk.md; the stash of a 1M conversation is 16.8 GB, not the 14.5 GB of the KV): the second request at 1M kills the server.** One request at 1M works. A second request that is not a prefix
extension (a new conversation) stashes the first one (KV 14.5 GB plus state) on top of the 45 GB and the cgroup
OOM-kills the server at the 54 GB limit (`oom-kill`, 54 GB peak). `IE_PROMPT_RAM_GB=0` did not prevent it. All 1M
server runs below are therefore a single request per server. At 512K the stash takes 3.5 s and works. The prompt-RAM
budget is not logged; with 54 GB limit, 45 GB RSS and 15 GB per 1M conversation no stash fits, so a 1M conversation
cannot be stored beside the model (the machine has 60 GB, the desktop needs the rest).

## 3. Cost of YaRN below the native length (teacher-forced, Δ-NLL)

Method of #73: prefill 32,768 or 131,072 tokens, the last 4,096 tokens through the megakernel teacher-forced; the
reference is the same text with `off`, the variant runs prefill and decode with YaRN. 95 % interval by block bootstrap
(64 tokens). Rule of `docs/backends/README.md`: upper end below 0.01 pass; mean below 0.01 and upper end above: undecided;
mean at or above 0.01: fail.

```
run                                              n  NLL ref NLL here     dNLL  95% low   95% up       KL  top1=
cpp-32768-4096-x2                             4096   0.2229   0.2237  +0.0008  -0.0046  +0.0058  0.00718  98.63
cpp-32768-4096-x4m1                           4096   0.2229   0.2237  +0.0008  -0.0041  +0.0055  0.00629  98.85
cpp-32768-4096-x4                             4096   0.2229   0.2275  +0.0046  -0.0030  +0.0119  0.01345  98.39
sh-32768-4096-x2                              4096   0.8795   0.8824  +0.0029  -0.0039  +0.0096  0.01946  96.90
sh-32768-4096-x4m1                            4096   0.8795   0.8795  -0.0000  -0.0047  +0.0047  0.01540  97.12
sh-32768-4096-x4                              4096   0.8795   0.8813  +0.0017  -0.0063  +0.0095  0.03388  95.58
cpp-131072-4096-x2                            4096   0.2513   0.2515  +0.0003  -0.0057  +0.0066  0.01537  98.34
cpp-131072-4096-x4m1                          4096   0.2513   0.2502  -0.0011  -0.0055  +0.0031  0.01360  98.41
cpp-131072-4096-x4                            4096   0.2513   0.2558  +0.0045  -0.0026  +0.0111  0.01916  98.10
sh-131072-4096-x2                             4096   0.7769   0.7800  +0.0031  -0.0043  +0.0100  0.02270  95.73
sh-131072-4096-x4m1                           4096   0.7769   0.7800  +0.0030  -0.0047  +0.0114  0.02337  96.19
sh-131072-4096-x4                             4096   0.7769   0.7910  +0.0140  +0.0038  +0.0242  0.03999  94.53
```

- x2 (mscale 1.0693): mean Δ ≤ +0.0031, four of four intervals contain 0, upper ends 0.0058 to 0.0100.
- x4 with the YaRN mscale 1.1386: **fails on shell at 128K** (+0.0140 [+0.0038, +0.0242]); on C++ +0.0045 / +0.0046.
- x4 with mscale 1.0: -0.0011 to +0.0030; the mscale alone is the difference between x4 (+0.0140) and x4m1 (+0.0030)
  on shell at 128K. The upper end 0.0114 there leaves it undecided. The 4,096-token intervals are wide (±0.007).
- Top-1 agreement with the native run 94.5 to 98.9 %, KL 0.006 to 0.040.

**Task benchmark** (`bench/threshold/`, pydash + boltons, 64K shared context, 600 retrieval questions A + 292 code
tasks B with the repositories' tests in the sandbox, greedy, thinking off, `w77`, default `IE_IDX_SHARE`, one run per
variant, paired). Variant: x4 with mscale 1.0 (the cheapest factor-4 variant by Δ-NLL), against off:

| run | family | n | acc % | b (lost) | c (gained) | McNemar p | diff pp [95 % Newcombe] |
|---|---|---|---|---|---|---|---|
| y-off | A | 600 | 93.3 | | | | |
| y-x4m1 | A | 600 | 92.8 | 8 | 5 | 0.58 | -0.5 [-1.8, +0.8] |
| y-off | B | 292 | 29.8 | | | | |
| y-x4m1 | B | 292 | 27.4 | 24 | 17 | 0.35 | -2.4 [-6.7, +1.9] |

Against the margins of #76 (A -3 pp, B -5 pp): A is inside; B is open (lower end -6.7 pp). The benchmark resolves about
2.5 pp on A and 7 pp on B, so a small loss on B is not excluded. Only the factor-4 mscale-1.0 variant was run; x2 has
not been run on the task benchmark.

## 4. Beyond the native length: needles

Text: a code corpus of our repositories and a few other projects (text only, `corp/code.txt`), prefix of the needed
length, 8 needles `# NOTE: the access code for vault <name> is <6 digits>.` at the fractions (k + phase) / 8 of the
document, then 8 questions with answers in one block (`Q: What is the access code for vault X?` / `A: <code>`).
**Teacher-forced**: prefill of everything but the last 191 tokens, then the block through the megakernel in one pass
(`IE_TEACHER_NLL`). A needle counts as a hit when all 6 answer digits are top-1; "NLL" is the sum of the 6 digits
(chance level 13.8 nats). Three phases (0.25, 0.5, 0.75; other codes and other depths) give 24 needles per length and
variant (1M: phases 0.5 and 0.25, 16 needles).

Hit rate over all needles of a variant, by distance of the needle to the question:

```
32K   off   n=24 hits 21/24 meanNLL   0.52 | 0-128K: 21/24 (0.5)
64K   off   n=24 hits 17/24 meanNLL   0.57 | 0-128K: 17/24 (0.6)
128K  off   n=24 hits 16/24 meanNLL   1.96 | 0-128K: 16/24 (2.0)
128K  x4    n=24 hits 21/24 meanNLL   0.99 | 0-128K: 21/24 (1.0)
250K  off   n=24 hits  2/24 meanNLL   3.13 | 0-128K: 2/12 (2.7) 128-256K: 0/12 (3.6)
250K  x4    n=24 hits  7/24 meanNLL   4.65 | 0-128K: 3/12 (4.2) 128-256K: 4/12 (5.1)
512K  off   n=24 hits 16/24 meanNLL   0.74 | 0-128K: 4/6 (0.5) 128-256K: 4/6 (0.7) 256-512K: 8/12 (0.8)
512K  x2    n=24 hits 15/24 meanNLL   1.69 | 0-128K: 3/6 (1.2) 128-256K: 5/6 (1.8) 256-512K: 7/12 (1.9)
512K  x4    n=24 hits 17/24 meanNLL   0.80 | 0-128K: 6/6 (0.2) 128-256K: 5/6 (0.3) 256-512K: 6/12 (1.4)
1M    off   n=16 hits  8/16 meanNLL   2.01 | 0-128K: 2/2 (0.1) 128-256K: 1/2 (1.2) 256-512K: 1/4 (2.2) 512-1074K: 4/8 (2.6)
1M    x4    n=16 hits  6/16 meanNLL   4.04 | 0-128K: 1/2 (1.9) 128-256K: 0/2 (4.9) 256-512K: 2/4 (5.2) 512-1074K: 3/8 (3.8)
1M    x4m1  n=16 hits  5/16 meanNLL   4.79 | 0-128K: 1/2 (0.5) 128-256K: 1/2 (3.3) 256-512K: 1/4 (8.9) 512-1074K: 2/8 (4.2)
```

Per run (columns: needle depth as fraction of the text, row 2 the distance to the question in K tokens; `*` = hit):

```
[32k] run                                      0.062     0.188     0.312     0.438     0.562     0.688     0.812     0.938   sum NLL  exact
depth fraction / distance to end (K)        30        26        22        18        14        10         6         2
n32k-off                              3.53      0.03   *  0.02   *  0.54   *  0.06   *  0.00   *  0.01   *  0.01   *      4.20    7/8
[32k-p25] run                                      0.031     0.156     0.281     0.406     0.531     0.656     0.781     0.906   sum NLL  exact
depth fraction / distance to end (K)        31        27        23        19        15        11         7         3
n32k-p25-off                          0.09   *  0.08   *  0.01   *  0.01   *  0.06   *  0.52   *  0.01   *  0.00   *      0.78    8/8
[32k-p75] run                                      0.094     0.219     0.344     0.469     0.594     0.719     0.844     0.969   sum NLL  exact
depth fraction / distance to end (K)        29        25        21        17        13         9         5         1
n32k-p75-off                          0.37   *  3.05      0.44   *  0.12   *  0.72   *  0.15   *  2.53      0.12   *      7.49    6/8
[64k] run                                      0.062     0.188     0.312     0.438     0.562     0.688     0.812     0.938   sum NLL  exact
depth fraction / distance to end (K)        60        52        44        36        28        20        12         4
n64k-off                              1.74      0.05   *  0.07   *  2.71      0.07   *  0.00   *  0.04   *  0.01   *      4.70    6/8
[64k-p25] run                                      0.031     0.156     0.281     0.406     0.531     0.656     0.781     0.906   sum NLL  exact
depth fraction / distance to end (K)        62        54        46        38        30        22        14         6
n64k-p25-off                          0.06   *  0.02   *  1.14      0.04   *  0.03   *  0.99      0.01   *  0.01   *      2.29    6/8
[64k-p75] run                                      0.094     0.219     0.344     0.469     0.594     0.719     0.844     0.969   sum NLL  exact
depth fraction / distance to end (K)        58        50        42        34        26        18        10         2
n64k-p75-off                          0.17   *  0.27   *  1.56      0.07   *  0.09   *  1.25      2.76      0.53   *      6.71    5/8
[128k] run                                      0.062     0.188     0.312     0.438     0.562     0.688     0.812     0.938   sum NLL  exact
depth fraction / distance to end (K)       122       105        89        73        57        41        24         8
n128k-off                             9.12      0.46   *  7.12      0.49   * 13.03      0.36   *  1.02   *  4.40         36.01    4/8
n128k-x4                              0.46   *  0.04   *  0.65   *  0.88   *  7.22      0.05   *  0.12   *  0.24   *      9.65    7/8
[128k-p25] run                                      0.031     0.156     0.281     0.406     0.531     0.656     0.781     0.906   sum NLL  exact
depth fraction / distance to end (K)       126       110        93        77        61        45        28        12
n128k-p25-off                         0.01   *  0.05   *  2.70      0.00   *  0.05   *  0.22   *  0.00   *  0.00   *      3.05    7/8
n128k-p25-x4                          0.03   *  0.37   *  0.30   *  0.04   *  0.03   *  0.27   *  0.01   *  0.00   *      1.05    8/8
[128k-p75] run                                      0.094     0.219     0.344     0.469     0.594     0.719     0.844     0.969   sum NLL  exact
depth fraction / distance to end (K)       118       101        85        69        53        37        20         4
n128k-p75-off                         1.19      0.04   *  2.40      0.67   *  0.19   *  0.02   *  3.47      0.04   *      8.02    5/8
n128k-p75-x4                          0.73   *  0.05   *  5.55      0.04   *  0.05   *  0.06   *  6.50      0.04   *     13.02    6/8
[250k] run                                      0.062     0.188     0.312     0.438     0.562     0.688     0.812     0.938   sum NLL  exact
depth fraction / distance to end (K)       234       203       172       140       109        78        47        16
n250k-off                             6.75      1.09      7.35      0.73      5.44      3.47      1.55      3.64         30.02    0/8
n250k-x4                              4.51      0.02   *  7.08      0.67   *  0.89      0.05   *  0.55   *  1.47         15.24    4/8
[250k-p25] run                                      0.031     0.156     0.281     0.406     0.531     0.656     0.781     0.906   sum NLL  exact
depth fraction / distance to end (K)       242       211       179       148       117        86        55        23
n250k-p25-off                         2.52      4.30      6.91      1.60      3.73      0.51   *  2.91      1.21         23.68    1/8
n250k-p25-x4                          1.01      0.31   *  8.90      0.74   *  1.62      1.30      2.65      0.35   *     16.87    3/8
[250k-p75] run                                      0.094     0.219     0.344     0.469     0.594     0.719     0.844     0.969   sum NLL  exact
depth fraction / distance to end (K)       226       195       164       133       101        70        39         8
n250k-p75-off                         1.56      2.44      6.94      0.95      1.05      3.00      0.12   *  5.24         21.30    1/8
n250k-p75-x4                          7.63      7.06      9.04     13.95     20.44      4.91      1.21     15.22         79.48    0/8
[512k] run                                      0.062     0.188     0.312     0.438     0.562     0.688     0.812     0.938   sum NLL  exact
depth fraction / distance to end (K)       487       422       357       292       227       162        97        32
n512k-off                             2.06      0.01   *  2.16      4.22      1.58      0.00   *  0.37   *  0.01   *     10.41    4/8
n512k-x2                              5.08      0.06   *  3.99      5.85      9.49      0.02   *  1.34      0.12   *     25.95    3/8
n512k-x4                              4.84      0.01   *  3.14      2.47      1.37      0.01   *  0.01   *  0.01   *     11.85    4/8
[512k-p25] run                                      0.031     0.156     0.281     0.406     0.531     0.656     0.781     0.906   sum NLL  exact
depth fraction / distance to end (K)       503       438       373       309       244       179       114        49
n512k-p25-off                         0.03   *  1.51      0.04   *  0.09   *  0.57   *  2.20      1.26      0.02   *      5.71    5/8
n512k-p25-x2                          0.25   *  0.11   *  0.01   *  5.31      0.22   *  0.69   *  0.20   *  0.01   *      6.80    7/8
n512k-p25-x4                          1.42      0.02   *  3.17      0.81      0.06   *  0.21   *  0.13   *  0.08   *      5.91    5/8
[512k-p75] run                                      0.094     0.219     0.344     0.469     0.594     0.719     0.844     0.969   sum NLL  exact
depth fraction / distance to end (K)       471       406       341       276       211       146        81        16
n512k-p75-off                         0.00   *  0.00   *  0.00   *  0.06   *  0.01   *  0.01   *  0.88      0.64   *      1.61    7/8
n512k-p75-x2                          0.37   *  0.00   *  0.01   *  1.53      0.02   *  0.08   *  1.67      4.08          7.76    5/8
n512k-p75-x4                          0.01   *  0.00   *  0.00   *  0.29   *  0.28   *  0.01   *  0.20   *  0.61   *      1.40    8/8
[1m] run                                      0.062     0.188     0.312     0.438     0.562     0.688     0.812     0.938   sum NLL  exact
depth fraction / distance to end (K)       975       845       715       585       455       325       195        65
n1m-off                               5.02      0.31   *  7.89      2.64      1.95      0.02   *  0.08   *  0.23   *     18.15    4/8
n1m-x4                                8.61      0.01   *  0.18   *  0.35   *  0.69   *  0.01   *  2.48      0.02   *     12.35    6/8
n1m-x4m1                              5.01      0.04   *  0.71   *  2.16     12.12      0.01   *  0.02   *  0.04   *     20.12    5/8
[1m-p25] run                                      0.031     0.156     0.281     0.406     0.531     0.656     0.781     0.906   sum NLL  exact
depth fraction / distance to end (K)      1007       877       747       617       487       357       227        97
n1m-p25-off                           0.20   *  0.05   *  4.09      0.50   *  5.44      1.34      2.31      0.03   *     13.97    4/8
n1m-p25-x4                            2.30      1.32      4.87     12.55     17.72      2.48      7.27      3.75         52.28    0/8
n1m-p25-x4m1                          7.69      3.24      3.88     10.67     22.19      1.39      6.52      0.97         56.56    0/8
```

**Greedy, chat format** (one request; the document, then the question; `--ctx` as needed, MTP):

| length | variant | codes found | prefill | decode | tokens per pass | pass |
|---|---|---|---|---|---|---|
| 512K (519.5K tokens) | off | 8 of 8 (8 separate requests) | 374 s = 1,389 tok/s | 700 tokens, 57.6 tok/s | 2.70 | 46.9 ms |
| 1M (1,039.5K tokens) | off | 7 of 8 (ember: 682564 for 682554) | 1,025 s = 1,014 tok/s | 900 tokens, 53.0 tok/s | 2.77 | 52.2 ms |
| 1M | x4 (mscale 1.1386) | 8 of 8 | 1,024 s = 1,015 tok/s | 900 tokens, 52.1 tok/s | 2.61 | 50.1 ms |

The 128K reference: 63.9 tok/s at 2.82 tokens per pass = 44.1 ms. The pass time grows 44 / 47 / 52 ms from 128K /
512K / 1M. MTP acceptance does not collapse with YaRN (2.61 against 2.77 tokens per pass at 1M; this is the vLLM #37435
failure mode, which does not occur: the MTP layer shares the table).

**Prefill of the teacher-forced tool** (`ie-gpu-gen`, `IE_SEQS=1`): 519,419 tokens in 343 s = **1,515 tok/s**;
1,039,348 tokens in 911 s (15.2 min) = **1,141 tok/s** (identical for all variants within 0.4 %). The server numbers
above are lower (checkpoint, chat template). Decode in the tool after the prefill: 31 tok/s (teacher-forced, one token
per pass, not a rate to compare).

## 5. Findings

1. **YaRN below 262K:** x2 costs nothing measurable; x4 costs through its mscale: with mscale 1.0 the Δ-NLL is
   within +0.003 and the task benchmark loses -0.5 pp (A) and -2.4 pp (B, not excluded either way).
2. **Beyond 262K, YaRN shows no consistent gain in this test.** 512K: off 16 / x2 15 / x4 17 hits of 24. 1M: off 8,
   x4 6, x4m1 5 of 16, and the answer NLL doubles with YaRN. The greedy chat run at 1M is the opposite (off 7/8, x4 8/8,
   one sample). The single numbers move by ±5 of 24 from text to text, more than any difference between variants.
3. **The test itself is weak.** Inside the native length the hit rate is 21/24 at 32K, 17/24 at 64K, 16/24 at 128K and
   **2/24 at 250K** (off), and 16/24 at 512K, 8/16 at 1M. It is not monotone in length, so the length is not what
   decides in this format; the text and the needle placement are. x4 gets 21/24 at 128K and 7/24 at 250K against 16 and
   2 for off, i.e. YaRN also helps inside the native length here. The chat-format greedy run (8/8 at 512K, 7/8 at 1M)
   is easier than the teacher-forced block. After the first digit of a code the NLL of the rest is near 0 (induction), so a
   miss means the needle was not found, not that it was misread.
4. **Who limits is not answered by this data.** Missing needles sit at every distance (at 1M with off: hits at 97K and
   1,007K, misses at 227K and 357K); there is no step at 262K as a RoPE-distance limit would give, and the 250K case
   shows the same failure inside the native length. That points to the selection (indexer top-512 of 4-token blocks) or
   the DeltaNet state rather than the rotation, but the indexer recall (research/33 C1 experiment 2) was **not
   measured**: the megakernel runs all 48 layers in one launch, the q of a QSA layer is overwritten by the next layer, so
   the measurement needs a new dump op in the program plus a dense reference over all positions per layer (K int8 for 1M
   tokens is 14.5 GB in pinned RAM) and a new analysis path. Not built.
5. **Time:** prefill 1M 15.2 min (tool) to 17 min (server), 512K 5.7 to 6.2 min; decode pass 44 / 47 / 52 ms at
   128K / 512K / 1M. These beat the estimates of research/32 (16-23 min, 65-80 ms).

## 6. Recommendation for phase 2

- **No YaRN mode per request yet.** Neither the retrieval tests nor the Δ-NLL show a gain beyond 262K that is larger than
  the noise, and x4 with the YaRN mscale fails the 0.01-nat rule at 128K. If a factor is wanted later, take mscale 1.0.
- **Allow `--ctx` beyond 262K with off** (the warning stays): 512K (8/8) and 1M (7/8) retrieve in the chat format.
- Before deciding on C2 (Jet-Long) or C3 (larger top-k): measure with the **chat-format greedy test and more needles and
  texts** (the teacher-forced block is too noisy) and build the indexer recall (C1 experiment 2). C3 is cheap to try
  (top-k 512 to 1024 at 512K and 1M with the same greedy test).
- Fix the 1M stash: a conversation that does not fit the RAM budget must not be stashed (today the server dies).
