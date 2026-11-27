# Register spills of k_mega (#66), 2026-10-28

Question: the megakernel spills registers (#63 note: 0.9 % decode lost on the server path since the device split).
Where do the spills sit, what do they cost, and does removing some of them bring the speed back?
Result: **no variant is faster on the server path (MTP); all three measured variants are slower. The code stays unchanged.**

Tools: `scripts/kernel-regs.py` (register profile per kernel from the code object), `llvm-objdump -d --line-numbers`
(the build already has `-gline-tables-only`), `scripts/isa-diff.py`. Model Coder IQ1_M, idle machine.

## Register profile of main (851f0fa)

| Kernel | VGPR | SGPR | VGPR spill | SGPR spill | Scratch (B) |
|---|---:|---:|---:|---:|---:|
| `k_mega<1>` | 192 | 107 | 56 | 93 | 328 |
| `k_mega<4>` | 192 | 107 | 49 | 97 | 296 |
| `k_mega<8>` | 192 | 108 | 126 | 135 | 440 |

(Before #63: VGPR spill 53 / 49 / 136, SGPR spill 106 / 110 / 176: the totals hardly changed, the placement did.)
SGPR spills go into lanes of three VGPRs (v189 to v191). Scratch includes 88 bytes for `Ctx`, which lives in memory
because its address goes to the `__noinline__` ops; that is not a spill. Spill slots start at offset 88.

## Where the spill instructions are

All ops except the matvec are `__noinline__` functions and have no spills of their own, apart from `op_qsa_prep` and
`op_dqk` (1 scratch store, 1 load, 2 `v_writelane`, 2 `v_readlane` each).

| Place | Spill instructions (static; `k_mega<1>` / `<4>` / `<8>`) | Hot or cold | Runs per pass |
|---|---|---|---|
| Loop head and prolog (`mega_kernel.h`: `init_ctx`, address sums hoisted out of the step loop) | stores 53 / 59 / 75, SGPR saves 47 / 48 / 52 | cold: once per launch | 1 |
| Calls of the noinline ops (`dispatch_qsa`, `dispatch_draft`) | saves 10 / 0 / 14, SGPR saves 34 / 34 / 22, restores 66 / 60 / 87 (SGPR 110 / 110 / 134) | warm: per non-matvec step | 610 steps |
| `op_qsa_prep`, `op_dqk` bodies | 6 each | cold | 12 (`qsa.prep`); `op_dqk` only with the DFlash drafter |
| Inlined matvec path (`run_phase`, `units_loop_q8`, `rows_loop_*`, `wide_loop`, `row_*`, `epilogue`, `resolve_unit`) | reload sites: VGPR 238 / 527 / 2,584, SGPR 54 / 1,180 / 2,271 | **hot**: per task and unit loop | 483 matvec steps |

Reload sites are counted per template instance (format x NB x loop), so only a part of them runs per task. In `<4>`
(MTP, 2 to 3 tokens per pass) the SGPR reloads dominate: `epilogue` 505, `resolve_unit` 320, `rows_loop_vectors` 114,
`units_loop_q8` 80, `row_q8` 40; VGPR reloads in `row_direct` 312, `wide_loop` 74.

Steps per pass (program recorded for `scripts/barprof.py`, 1,093 steps, without MTP ops): matvec 483 (`gr.up`,
`gr.down` 97 each; `router`, `moe.shared`, `moe.gate_up`, `moe.down` 48 each; `dn.in`, `dn.out` 36 each; `qsa.in`,
`qsa.out` 12 each; `head.out` 1), other 610 (`mix`, `combine` 97 each; `moe.act`, `moe.sum`, `moe.copy`, `host.wait`,
`host.y` 48 each; `dn.conv`, `dn.recur`, `dn.norm` 36 each; `qsa.prep`, `qsa.score`, `qsa.topk`, `qsa.attn`,
`qsa.merge` 12 each, rest 8). Per token: divide by the tokens per pass (2 to 3 with MTP, 1 without).

Cause of the SGPR pressure: the compiler hoists address sums (`bar + 16`, `P + 20`, kernel argument base + 64,
`gridDim.x * 16 - 16` and similar, about 47 values of 64 bit) out of the step loop into SGPRs of their own, so the
matvec loops are left with too few.

## Variants (register profile; VGPR 192 in all; VGPR spill / SGPR spill / scratch)

| Variant | `<1>` | `<4>` | `<8>` |
|---|---|---|---|
| main | 56 / 93 / 328 | 49 / 97 / 296 | 126 / 135 / 440 |
| start, acc derived from `bar` | identical | identical | identical |
| flags (`arr`, `nobar_all`, stamp) read from `P` per step | 56 / 95 / 328 | 38 / 94 / 264 | 124 / 135 / 440 |
| `step_end` `__noinline__` | 56 / 78 / 360 | 52 / 77 / 344 | 118 / 126 / 472 |
| **v5** = both above | 53 / 78 / 328 | 37 / 77 / 264 | 116 / 126 / 440 |
| **v6a** empty `asm("" : "+s"(P), "+s"(bar))` at the head of each step | 56 / 76 / 328 | 56 / 68 / 328 | 131 / 86 / 440 |
| v5 + v6a | 54 / 64 / 328 | 52 / 162 / 312 | 111 / 72 / 440 |
| v6a with `asm` also on `tasks`, `ops` | build error (`loop not unrolled` in `matvec_wide.h`, `-Werror`) | | |
| waves per EU 7 (216 VGPR) | 37 / 93 / 248 | 79 / 97 / 392 | 87 / 139 / 328 |
| waves per EU 6 (240 VGPR) | 0 / 93 / 104 | 41 / 97 / 264 | 103 / 145 / 440 |

v5 reduces the LDS of the kernel by 8 bytes (32,024 to 32,016).

**Wave limit rejected without a GPU run.** The block count is `2 * cus - free_cus()` (`Mega::init_kernels`), independent
of the occupancy (the HIP calculation is information only). A block has 16 waves = 4 per SIMD of a WGP; two blocks per
WGP need 8 waves per SIMD, that is at most 192 VGPRs. With 216 or 240 VGPRs only one block per WGP is resident (16 of
32): the barriers would hang the GPU. More VGPRs would need `IE_MEGA_WAVES` below 16, which changes the order of the
reductions and with it the numbers. The SGPR spills do not depend on the wave limit at all (93 / 97 / 139 to 145).

## Speed (v5 and v6a against main; golden identical)

Golden: `golden.sh compare` (GPU plan, CPU plan) and `golden-server.sh compare` identical for v5 and v6a.
Three rounds, interleaved per round (main, v6a, v5), one server start per build and round (MTP, `--ctx 32768`), no
build and no other GPU job. Raw output in `/var/tmp/claude/w66-meas/`.

Server with MTP, decode tok/s (`scripts/bench-set.py`):

| | main | v6a | v5 |
|---|---|---|---|
| code | 87.8, 89.0, 89.0 | 88.2, 88.3, 88.2 | 87.4, 87.4, 87.3 |
| text | 64.2, 64.8, 64.8 | 63.3, 63.3, 63.2 | 62.5, 62.5, 62.5 |
| prefill 13K | 1,758, 1,760, 1,761 | 1,760, 1,760, 1,761 | 1,760, 1,759, 1,760 |
| decode after 13K | 61.9, 62.1, 64.9 | 66.1, 60.0, 60.8 | 58.9, 62.3, 63.2 |

ie-gpu-gen with DFlash2 (`scripts/bench-decode.sh 1`, 200 tokens, decode tok/s):

| | main | v6a | v5 |
|---|---|---|---|
| csv | 92.2, 92.2, 92.2 | 88.8, 88.8, 88.8 | 88.7, 88.7, 88.7 |
| rust | 80.9, 80.9, 80.9 | 77.9, 78.0, 78.0 | 77.8, 77.7, 77.7 |
| explain | 58.2, 58.3, 58.3 | 56.1, 56.1, 56.1 | 56.0, 56.1, 56.1 |

The spread within a build is 0.2 tok/s or less (except decode after 13K, which scatters by 3 in all builds). Both
variants are slower on the server path: code -0.8 % (v6a) and -1.8 % (v5), text -2.3 % and -3.4 %. DFlash2 -3.7 %
in all three prompts, but see the control series: this path turned out not to be reproducible.

## Control series: is it the build? (second series, same protocol)

Both variants lost the same 3.7 % on DFlash2, which looked like a common build difference. Checked:

- `CMakeCache.txt` of main, v5 and v6a: same build type (`RelWithDebInfo`, `-O2 -g`), same `IE_CHAIN_WAVES=8`,
  `IE_MARCH=znver5`, same compiler; only the source directory differs. `-gline-tables-only` is a public compile option
  of `ie_gpu` and is on in all builds. The `-cuid` of `mega.hip` is the same in all builds (`879d5cec3d7eb9fd`).
- `isa-diff` main against v6a: 171 functions each, only the three `k_mega<NB>` differ (v5 additionally `op_route` and
  `op_host_wait`, which sit behind `k_mega` in the file and move). No other kernel changes.
- Control build of unchanged main from the worktree (other source path): `isa-diff` against main reports
  `all kernels identical`. The edit "start, acc derived from `bar`" alone is also identical in machine code.

So the build is not the cause. Second series, three rounds interleaved, main / control (identical code) / v2 (flags
read from `P` per step, profile `<4>` 38 / 94 / 264), same protocol:

| | main | control | v2 |
|---|---|---|---|
| server code | 89.1, 89.1, 89.0 | 89.1, 89.1, 89.2 | 88.8, 87.1, 88.4 |
| server text | 64.9, 64.8, 64.8 | 64.9, 64.9, 64.9 | 64.5, 64.5, 64.5 |
| prefill 13K | 1,760, 1,759, 1,759 | 1,760, 1,761, 1,759 | 1,760, 1,761, 1,753 |
| DFlash2 csv | 92.2, 88.9, 89.7 | 92.3, 91.4, 89.1 | 88.3, 91.6, 91.5 |
| DFlash2 rust | 81.0, 80.9, 81.1 | 81.1, 78.7, 81.0 | 80.3, 78.1, 78.0 |
| DFlash2 explain | 56.6, 56.6, 56.6 | 58.3, 56.7, 56.7 | 57.8, 57.8, 56.2 |

- Server path: main and control agree within 0.1; v2 is 0.3 to 1 % slower in code and 0.5 % in text.

### DFlash2 splits in two groups

- **DFlash2 with one run per round is not reproducible:** main and the identical control give 88.9 to 92.3 (csv) and
  56.6 to 58.3 (explain) in the same series. The values fall into two groups (about 89 and about 92 in csv) whatever the
  build, so the 3.7 % of the first series cannot be read as a code effect; DFlash2 decides nothing here. The server
  path with MTP is the reliable measurement (spread <= 0.2 within a build).
  `bench-decode.sh` with one run per round decides nothing for differences below about 4 %.

## Result

- The spills sit in the inlined matvec loops (hot) and, as one-off saves, in the prolog (cold). The ops themselves
  do not spill.
- Fewer spill instructions do not mean more speed: the variants remove 20 to 30 SGPR spills in `<4>` and lose
  0.5 to 3.4 % on the server path. A count of spills says nothing about the dynamic cost; the placement of the loops (unrolling,
  scheduling) decides, as already seen in #63. v5 + v6a shows how unstable it is: SGPR spills in `<4>` rise to 162.
- The 0.9 % of #63 is not recovered by changing the spill count. The code stays unchanged; the wave limit is not
  possible with 32 blocks of 16 waves.
