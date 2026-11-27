# 2026-10-23: Verify pass for up to 4 tokens in the megakernel

Own measurement. `engine/gpu/mega.hip`, tool `ie-gpu-gen --mega --batch-check N` or `--batch-bench N`.
RAM 32 GB, desktop on the same GPU, 32 blocks × 16 waves.

## Setup

- `forward(tokens, nb)` computes nb ≤ 4 tokens at pos … pos + nb − 1 in one kernel, `commit(m)` accepts the first
  m. All steps have a token index; buffers per token at a fixed stride.
- Matvecs: every sub-block is unpacked once and multiplied with nb vectors (kernel `k_mega<4>`); for
  single tokens `k_mega<1>` remains.
- KV cache, indexer keys, pool: by position, rejected tokens are overwritten later. Conv rings of
  DeltaNet and PLE with BMAX more places.
- DeltaNet without state copies (principle ReplaySSM, arXiv 2610.13778): the stored state lags one pass
  behind. Each pass first commits the m accepted tokens of the previous one (from their δ, k, α) and reads
  Sᵀk_b and Sᵀq_b of the new tokens in the same sweep; their δ and outputs follow in closed form from the dot products k_i·k_b,
  k_i·q_b. One read and one write sweep per pass as before.
- Experts: the CPU picks the top-10 per token, union (≤ 40) with a mask per token; the group tasks have
  their units at run time (n_u).

## Correctness

Prompt (33 tokens) first token by token, then in passes of up to 4 tokens, taking 1, 2, 3, 4 accepted in turn
(the rest discarded and recomputed), 14 passes, 53 comparisons:

| | Top-1 equal | largest relative deviation of the logits |
|---|---:|---:|
| float matvecs (`IE_MEGA_F32=1`) | 53 / 53 | 1.0e-6 |
| int8 | 53 / 53 | 0.11 (rounding of x flips, as against the CPU: 0.16) |

B = 1 against CPU (float): 4.85e-7.

## Speed

Measurement mode without reloading (`IE_FAKE_EXPERTS`), 400 positions, all tokens accepted:

| B | ms per pass | Tokens/s (decode) at full acceptance |
|---|---:|---:|
| 1 | 22.9 | 43.6 |
| 2 | 31.3 | 64.0 |
| 3 | 37.0 | 81.2 |
| 4 | 41.7 | 95.8 |

B = 1 after the rework as before (44.1–44.2 tok/s decode). On the way: the DeltaNet recurrence in one loop (commit
and read) waited for latency on every row (1.83 ms per token); loading 8 rows ahead 1.38 ms, plus each column
distributed over 2 lanes (384 instead of 192 waves): **0.83 ms** (before 0.91).

Time per pass at B = 4 (42.6 ms), largest items against B = 1:

| Step | B = 1 | B = 4 |
|---|---:|---:|
| moe.gate_up | 3.33 | 7.43 (union of the experts) |
| dn.in | 3.94 | 5.94 |
| moe.down | 2.04 | 5.21 |
| dn.recur | 0.83 | 3.50 |
| combine | 0.80 | 2.92 |
| head.out | 1.70 | 2.58 |
| qsa.attn | 0.52 | 1.97 |
| host.wait | 0.40 | 1.19 |

Occupancy: HIP reports for `k_mega<1>` with 182 VGPRs only one block per WGP, for 185 and 192 VGPRs two; 32 blocks
run with both without abort. The block count is therefore fixed at 2 × WGPs. `k_mega<4>` spills 380 B (scratch).

## Addendum: combine with all sums at the same time

`combine` computed the RMS per (token, stream) one after the other with the whole block (16 block sums at 4 tokens).
Now wpc = 16 / (nb · hc) waves per sum, all at the same time. Profile: B = 4 2.94 → 1.92 ms, B = 1 0.80 → 0.69 ms.
A/B against the previous state (same build options, `IE_CHAIN_WAVES=8`), 3 rounds per version, ms per pass:

| B | before | after |
|---|---:|---:|
| 1 | 23.05–23.10 | 22.98–22.99 |
| 2 | 31.29–31.33 | 31.30–31.36 |
| 3 | 36.69–36.74 | 36.55–36.56 |
| 4 | 41.62–41.74 | 40.89–40.98 |

Careful with the A/B: a fresh build takes `IE_CHAIN_WAVES=12` (default in CMakeLists), the working build has 8
in its cache; with 12, B = 1 is 1.3 ms slower.

## Addendum: DeltaNet recurrence with k/q in the LDS

For each row of the state every wave read 2·nb values of k_b/q_b and m old k_i from L2 (same address for all
lanes, hence latency-bound). Now each block loads the vectors of its 2 heads once into the LDS (3 KB at
4 tokens). Profile dn.recur: B = 1 0.83 → 0.65 ms, B = 4 3.50 → 1.70 ms. Pass time (2 runs, ms):

| B | before | after |
|---|---:|---:|
| 1 | 22.99 | 23.06–23.11 |
| 2 | 31.30 | 30.01–30.08 |
| 3 | 36.55 | 35.00–35.02 |
| 4 | 40.91 | 38.81–38.82 |

float check unchanged (verify pass 8.7e-7, top-1 53/53; against CPU 5.6e-7).

## Addendum: x of the matvec phase in the LDS (passes with several tokens)

Diagnosis at B = 4 (profile 39.8 ms per pass), quick tests with deliberately wrong results:

| Test | Pass |
|---|---:|
| as built | 39.8 ms |
| load x only once (same values for all 4 tokens) | 32.7 ms |
| 4 load instructions as built, but all to the same address | 33.2 ms |
| weights with nontemporal load instructions | 39.8 ms |
| fewer lanes per row (8 or 4, more rows share x) | 43.1 / 47.2 ms |
| k_mega<4> with less register spilling (recurrence for 1 token) | matvecs equal |

So it is the data, not the number of instructions: x for 4 tokens (with 2560 inputs 12.8 KB including
metadata) does not stay in L0 next to the weight stream. Now each block copies x once per phase into the LDS
if all int8 tasks of the phase read the same x and it fits in the existing buffer (17.7 KB): dn.in,
qsa.in, router, moe.gate_up, head.out, gr.up and most drafter phases; not dn.out, qsa.out (6144
inputs), gr.down (10240) and moe.down (x per expert).

A larger buffer (55 instead of 32 KB LDS per block, for dn.out and qsa.out) **hung in 4 of 4 runs**
("no message from the router", desktop on the same GPU), hence not used.

Pass time (2 runs, ms; `IE_MEGA_NO_STAGE=1` switches the staging off):

| B | without staging | with staging |
|---|---:|---:|
| 1 | 23.03 | 23.04–23.08 |
| 2 | 28.54–28.55 | 27.72–27.76 |
| 3 | 33.58–33.60 | 32.10–32.12 |
| 4 | 37.29–37.31 | 35.06–35.11 |

Without staging B = 4 is 37.3 here instead of 38.8 ms as in the addendum before; only the register allocation
had changed (spill 136 → 120 B per lane). Drafter kernel 5.3 → 5.0 ms.

## Addendum: output no longer bit-identical, adaptive length without measurement

- int8: speculation and single steps diverge from byte 491 of the CSV example. There `"""` and another
  doc line are nearly tied. Reason: B = 1 and B = 4 sum in a different order (expert union,
  combine). Before, it happened to match. A fixed summation order in combine cost 0.4 ms and is not enough, hence
  discarded.
- float (`IE_MEGA_F32=1`): with and without speculation **equal**, 119 tokens in 32 passes (3.72 tokens per pass).
- Check `--batch-check` now with passes of 4, 3, 2, 1 tokens in turn: float 9.7e-7, top-1 45/45.
- The adaptive draft length took the measured pass times along on a sliding basis. With reloading (32 GB) it thereby chose
  narrow passes (widths 8 / 13 / 14 / 12) and reached 2.55 tokens per pass, 16.1 tok/s (decode). With fixed
  pass times it was 3.53 tokens per pass, 32 of 34 passes full. One expensive pass deters permanently, but
  reloading hardly depends on the width. The sliding measurement is therefore only used with `IE_DFLASH_EMA=1`.
