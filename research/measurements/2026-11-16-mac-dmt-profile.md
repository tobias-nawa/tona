# Mac: an expert profile for DM-T, carried from the Coder's or counted (2026-11-16)

Decision this serves: which profile DM-T's arena ranks its experts by, for the A/B of DM-T with and without the arena
(the A/B without a profile: decode -47 to -57 %, 2026-11-16-mac-dmt-arena-ab.md).

## Plan

- Carry: `tools/coder_requant/profile_map.py` matches each DM-T router row (BF16) byte for byte to a row of the
  Coder shard bba88856 and takes that expert's count from `toolchain/profiles/coder-iq1_m.bin` (counted on the PC
  2026-10-23). DM was cut from the older shard 5348543e (25 % of the experts differ), so a share stays unmatched.
- Count: tona-server with `TONA_METAL_ARENA_GB=auto`, `TONA_EXPERT_PROFILE_OUT`, `TONA_DETERMINISTIC=1`, MTP-164,
  the six prompts of scripts/make-profile.sh, 250 tokens each, temperature 0 (set 1); then six other coding prompts
  (set 2, held out). About 3 min per set.
- Comparison: the share of set 2's uses that the top k experts of a layer by each profile cover (k = 80, 104, 130 of
  164; 104 is what the arena holds at its budget of 14 GB). Expectation: counted above carried, carried above
  random; no number expected beforehand (first count on the Mac).

## Results

Carried: 21.0 % of DM-T's experts have no partner in the new shard (count 0); the matched ones carry 70.5 % of the
Coder profile's uses.

Share of set 2's uses (834,720 routings) in the top k of a layer:

| k of 164 | counted (set 1) | carried | mixed 1:1 | mixed 1:0.5 | random | set 2 on itself |
|---|---|---|---|---|---|---|
| 80 | 75.9 % | 68.3 % | 74.8 % | 75.6 % | 49.6 % | 80.8 % |
| 104 | **85.2 %** | 75.6 % | 84.3 % | 85.1 % | 63.9 % | 89.6 % |
| 130 | 93.2 % | 82.7 % | 92.7 % | 93.1 % | 79.7 % | 96.0 % |

**The counted profile wins**; carrying adds nothing in a mix (the unmatched fifth weighs it down). The profile for
the A/B is the sum of sets 1 and 2 (`work/profiles/dmt-12.bin`, local: DM-T is not published). With it the arena
holds 5641 of 7872 experts, and `TONA_DETERMINISTIC=1 tona-check` (64 tokens) is byte-identical to DM-T without the
arena.
