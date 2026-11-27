# Speed after the code cleanup (2026-10-27)

Question: does the cleanup (English, gates: files ≤ 500 lines, functions ≤ 80 lines, CCN ≤ 15, ≤ 6 parameters) cost
speed? Three builds, interleaved per round so clock and page-cache drift hit all of them alike:

| Build | State |
|---|---|
| old | `main` before the cleanup (928a144) |
| base | cleanup branch with host code of the megakernel, matvec/ops/forward, prefill/GEMM, server, tools split |
| megadev | base plus the device code of the megakernel split into headers (op dispatch per group) |

Idle machine (no builds, no other GPU jobs; desktop on the same card). Model Coder IQ1_M, all experts pinned (64 GB
RAM), `OMP_NUM_THREADS=6`, expert profile `work/profile/experts-coder.bin`.

## ie-gpu-gen with DFlash2 (`scripts/bench-decode.sh`, 200 tokens, one run per round)

Decode tok/s per round (three rounds; the first old round ran on a cold page cache):

| Prompt | old | base | megadev |
|---|---|---|---|
| csv | 90.1, 92.2, 92.2 | 92.6, 92.4, 92.2 | 91.5, 91.4, 91.6 |
| rust | 80.0, 80.7, 80.8 | 81.0, 80.9, 81.0 | 80.3, 80.2, 80.2 |
| explain | 57.7, 58.2, 58.2 | 58.4, 58.4, 58.3 | 57.9, 57.8, 57.8 |

Prefill, 13,328-token prompt (`scripts/bench-long-code.txt`), `--prefill --dflash`, tok/s: old 1,617 / 1,617 / 1,617,
base 1,617 / 1,619 / 1,617, megadev 1,585 / 1,616 / 1,619.

Note: without a drafter `ie-gpu-gen` does not take the prefill path; the prompt then runs token by token
(41 tok/s, 13,344 launches). A first prefill series without `--dflash` measured that by mistake.

## ie-server with MTP (production path)

`scripts/bench-set.py` (11 code, 3 text prompts, up to 400 tokens) and `scripts/bench-openai.py URL x 1 long`, one
server start per build and round, `--ctx 32768`.

| | old | base | megadev |
|---|---|---|---|
| decode code | 88.2, 89.4, 89.3 | 89.5, 89.4, 89.4 | 87.8, 87.7, 87.7 |
| decode text | 64.1, 64.8, 64.7 | 64.8, 64.8, 64.8 | 64.1, 64.1, 64.1 |
| prefill 13K | 1,756, 1,753, 1,753 | 1,753, 1,753, 1,752 | 1,753, 1,754, 1,754 |
| decode after 13K | 63.8, 64.3, 59.5 | 63.1, 62.8, 64.7 | 59.1, 64.5, 60.8 |

## Result

- **base equals old** in decode and prefill (within 0.2 tok/s; the first old round is the cold start). The split of
  host code, matvec/ops/forward, prefill/GEMM, server and tools costs nothing.
- **megadev is slower in decode:** −1 % with DFlash2, −1.9 % code and −1.1 % text with MTP, far outside the spread
  (≤ 0.2). Prefill is unchanged. The device-code split goes back for rework (bisect of its commits); it is not merged
  in this form.
- Decode after the 13K prompt scatters by ±3 tok/s between rounds in all builds; one run per round says nothing there.

## Rework of the device split (#63, 2026-10-28)

Bisect of the device-split commits, then fixes, each measured on both paths against the parent 637ba6c (equal to
base). Server path with MTP, `scripts/bench-set.py`, decode tok/s, rounds interleaved:

| Build | Code | Text |
|---|---|---|
| 637ba6c (parent) | 88.8 | 64.4 |
| headers only (6df0bae) | 88.9 | 64.3 |
| translate and split ops (5b35093) | 87.5 | 64.0 |
| dispatch chain, argument struct (f967f27) | 87.0 | 63.6 |
| **final (3b354c5)** | **88.0** | **64.1** |

With DFlash2 (`scripts/bench-decode.sh`) the final tree equals the parent within spread (csv 91.8 vs 91.9, rust 80.5
vs 80.5, explain 58.0 vs 58.0). Prefill is unchanged (1,725 vs 1,726).

- The largest single cost was `op_dn_recur`: split into helpers that pass structs and arrays, the compiler emitted
  half the vector FMAs (64 instead of 128 `fma.v4f32`). The body is back in its original shape with two small
  extractions; this recovered the DFlash2 path completely.
- The dispatch chain compiles to the same code as the single switch (identical instruction counts).
- The remaining 0.9 % (code) on the server path does not sit in one construct. Only the original 11 kernel parameters
  together with the original op bodies bring the register profile and the speed back; every other combination lands
  between 87.0 and 88.3. Field order, copies into locals and explicit parameters for the matvec case change nothing.
  The 11-parameter form breaks the gates, so the loss is accepted; k_mega spills already in the parent (#66), which
  is the place to win it back.
