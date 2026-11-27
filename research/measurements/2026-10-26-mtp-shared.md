# MTP steps of several conversations in one pass

Own measurement, 2026-10-26. `mtp_step_multi`: one token per conversation after its last MTP entry, all in
one pass of the MTP head (multi-conversation program as with the target: QSA steps and selection per slot). The
generator runs the chains in lockstep (`IE_MTP_MULTI=1`).

## Equality

`IE_MTP_MULTI_CHECK=1` recomputes every slot afterwards on its own: 1,600 steps, 0 different tokens, |Δq| = 0,
|Δh| = 0 (bit-exact). The first attempt deviated: the routing of the MTP head computed only the last row (one conversation
needs only that one), fixed.

## Cost of a step (balanced over the passes of two conversations, bench-parallel 2 clients)

| | fixed (catch-up twice) | per double step | per single step | chain length (sum) |
|---|---:|---:|---:|---:|
| separate | 3.02 ms | 2.18 ms (2 × 1.09) | 1.10 ms | 5.4 |
| shared | 3.18 ms | 1.71 ms | 1.07 ms | 7.0 |

Profile of a single step (`IE_MTP_PROF`, 995 µs): head 343 µs (shortened LM head, ~49K rows), qsa.in 140,
eh_proj 103, qsa.out 62, gate_up 58, down 33. The second row still costs +57 %: the matvecs compute per row,
the MoE reads the experts of both rows.

## Speed (two clients, decode)

| | total | per client | accepted per ms |
|---|---:|---:|---:|
| separate | 90.6 / 90.7 tok/s | 47.8–48.8 | 0.100 / 0.101 |
| shared | 89.3 / 89.2 tok/s | 47.1–48.9 | 0.101 / 0.101 |

No gain: the double step saves 0.47 ms, but the chains get longer (the other conversations enter the rate with their
last pass, not with the block already drafted). Default off, stays as an option.

One conversation alone unchanged: code 90.0 tok/s decode, text 64.9 (same tokens).

Further levers: catch-up of both conversations in one pass (2 × 1.5 ms fixed), the chain rule in lockstep with
the current block of the other; the second row cheaper in the head (matvec with several rows over weights read once).
