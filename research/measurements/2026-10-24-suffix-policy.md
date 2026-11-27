# Server suffix policy simulated, 2026-10-24, own measurement (simulation)

Tool `ie-sfx-policy`: the rule from `Generator::generate` (draft from the history from 8 identical tokens on,
`server/suffix`) on the last 40 sessions of the agent corpus (1.2 M output tokens). Acceptance as in
2026-10-22-suffix-sim: the model produces exactly the recorded output. Drafter passes assumed at 2.6 tokens,
time model like ie-spec-sim (bandwidth, measured expert union u(b)).

The corpus had tool calls in JSON format (code with `\n`, `\"` escaped). Our template emits parameters
raw in XML format (`<parameter=old_string>`); converted: `corpus-xml.txt`. Share from the history 32.2 → 35.0 %.

## Longer verify passes for suffix hits (pass up to 4 tokens today)

| Draft at most | Share history | Tokens per history pass | Tokens per pass overall | Time model |
|---|---:|---:|---:|---:|
| 3 (today) | 35.0 % | 3.30 (67 % full) | 2.81 | 108.9 |
| 7 | 35.0 % | 5.13 | 3.09 | 113.5 (+4 %) |
| 15 | 35.0 % | 6.84 | 3.25 | 114.2 (+5 %) |

(Rows 7/15 on the JSON corpus; XML same order of magnitude.) An attempt with `MEGA_BMAX = 8`: the kernel builds (192 VGPRs,
568 bytes scratch), but the result is wrong (teacher text NLL 1.485 instead of 0.961): places with a fixed assumption of 4 (drafter
block among others). **Deferred:** +4–5 % in the model against a rework at ~80 places.

## Minimum match length

| Minimum length | 4 | 6 | **8** | 12 | 16 |
|---|---:|---:|---:|---:|---:|
| Share history | 55.0 % | 42.5 % | 35.0 % | 25.6 % | 20.2 % |
| Time model | 105.3 | 108.5 | **108.9** | 107.6 | 106.6 |

8 (today's default) is the optimum.

## Tool results without line numbers (AgSpec "emission form")

Read results appear as `     1\tline`; idea: additionally index them without numbers. Simulated: 128,930 against
128,975 history passes, no difference. The 40 test sessions contain only 59 Edit calls: the corpus is mostly
conversation. Not built in; check again with edit-heavy sessions on the real model.
