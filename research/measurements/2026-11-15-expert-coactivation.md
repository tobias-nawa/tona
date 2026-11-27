# Expert co-activation as a predictor (#154): plan

Question: can statistics over the experts routed so far predict the experts to come, (vertical)
from (layer l, expert X) to (layer l+d, expert Y) for d up to 20, and (horizontal) from one predicted expert of a
layer to its usual and unusual partners in that layer? The value of the vertical chain is lead time: the next-layer
lookahead (router of layer l+1 on the residual of layer l, recall and precision 69 % on the Mac,
[2026-11-15-mac-coder-arena.md](2026-11-15-mac-coder-arena.md)) leaves about one layer to hide an SSD read, a chain
with d >= 5 would hide it fully. Research: [../55-expert-coactivation.md](../55-expert-coactivation.md).

## Data

- What the trace files hold (read in `engine/gpu/mega_select.cpp`): `TONA_ROUTE_TRACE` the union of experts per layer
  and pass (no per-token detail, useless here); `TONA_ROUTE_TRACE_P` per layer and pass, per token, the 256 router
  logits of the layer and the 256 logits of the lookahead for the next layer (so top-10 per token and the lookahead
  of the server are both reconstructable); `TONA_ROUTE_LOG` per token and layer "position layer experts" in rank
  order (gives the positions, so the passes can be cut into requests). Decode passes only: the prefill does not write
  either file.
- Existing: `/var/tmp/claude/148/res/s1.tracep` (2026-11-15, 14 bench-set prompts, 1,122 passes, 5,474 token rows
  including rejected drafts, 4,087 tokens). It has no positions, so no workload split. Used to build and check the
  tool only; no numbers from it are reported as results.
- New (one GPU job, queue label `154 coactivation trace`): the build `tiers` (main 4259191, route log and logit trace
  as above), 26 GB class (routing does not depend on the tier), MTP on (verify passes are what we need), greedy,
  `TONA_ROUTE_LOG` and `TONA_ROUTE_TRACE_P` on, `tools/coactivation_drive.py`: **codeA** 11 bench-set code prompts, **codeB**
  10 new code prompts, **text** 10 new prose prompts (English and German), **long** two requests of about 20k
  tokens of context (source code, research notes), decode of 300 tokens. Roughly 12,000 decode tokens, 1.3 GB trace
  in `/var/tmp/claude/154/res/`, a few minutes of GPU after the prefill of the long requests. Why this size: the
  question needs per-token top-10 per layer on at least two disjoint workloads of a few thousand tokens each
  (tables of 256 x 256 per layer pair are counted from them), nothing else. No VRAM pressure.

## Predictors and metrics

All sets are per token and layer; truth T = the 10 experts of the token at the target layer. Recall = |S and T| / 10,
precision = |S and T| / |S|. "Cold" = the part of T and S outside the **hot set** of the layer (the 85 most used
experts of the layer in the counting set, the VRAM slots of the 16 GB card): hot experts are resident, so a cold
prediction is the one that saves a read. Verify-pass numbers: the union of the predictions of the tokens of one pass
against the union of the truth of that pass (what a prefetch must actually fetch).

1. **Marginal** (control): the top-K experts by frequency of the target layer, no conditioning. Everything below must
   beat this, or it only rediscovered the skew.
2. **Lookahead** (a): the server's prediction for l+1 from `TRACE_P`, top-K by logit. Only d = 1. Cross-check against
   the Mac's 69 % / 69 %.
3. **Vertical chain** (b): `P(Y at l+d | X at l)` counted per token over the 10 experts of the source layer, tables
   per (l, d), d = 1..20. Predicted set: the K experts with the highest mean conditional probability over the
   token's 10 source experts (K sweep); and the rule form: all Y with `P(Y|X) >= theta` for some routed X with
   support >= 20, theta in {0.5, 0.7, 0.9} (the "90 % confidence" of the ticket).
4. **Horizontal partners** (c): `P(Y|X)` within a layer; (i) completion: seed = top-m of the lookahead (m = 3 to 8),
   fill up to K by the partner score, against the lookahead's own top-K at equal K; (ii) pruning: from the
   lookahead's top-24 drop experts with a low partner score given the seed; (iii) the share of pairs that "almost
   never" occur together (lift < 0.2 with enough support) and how many pairs are reliable.
5. **Combinations** (d): rank fusion (lookahead + chain + partners; weights chosen on the counting set), at
   d = 1 against the lookahead, at d >= 2 against the chain alone.
6. **Stability**: count on codeA, test on codeB (same workload, other prompts); count on code, test on text and the
   reverse; count on the short requests, test on the long ones; counting on the test workload itself as the upper
   bound. Table sizes: dense (48 x 20 x 256 x 256 counts, 16 bit) and sparse top-N partners per expert.
7. **Online** (decay): tables updated per token with decay while the test workload streams (half-lives 500 and
   5,000 tokens), starting from the shipped table of the other workload and from empty.

## Decision thresholds (set before the replay)

Test sets are always other prompts than the counting set. Numbers are cold recall at equal precision unless said.

- **Vertical chain: build** if at d >= 5 (and K chosen so that precision >= 50 %) the cold recall is >= 50 % and
  >= 15 points above the marginal control, measured on the cross-workload test (counted on code, tested on text) and
  on the verify-pass unions. A chain at d = 10 to 20 with >= 40 % cold recall at >= 50 % precision would justify a
  prefetch with SSD lead time by itself. Below 35 % at d = 5: no build, record the curve.
- **Horizontal completion/pruning, or fusion: build** if at d = 1 it lifts recall by >= 5 points at equal precision
  (equal K) over the lookahead alone, on the per-pass union at the 32 GB and Mac budgets. Less than 2 points: no
  build.
- **Shipped table or online**: ship a table only if the cross-workload loss against the in-workload table is <= 3
  points of recall at the chosen K; otherwise the table must be learned online (with decay), and the shipped table
  serves only as the start.
- All of these are replay results (no speed claim). A build would go through its own plan and an A/B
  (`scripts/ab.py`).

## Expectation (with tolerance, source)

No earlier measurement of cross-layer ID chains on this model exists (the profile counts reads per layer, not
transitions), so these are estimates from the skew: the replay of
[2026-11-15-expert-tiers.md](2026-11-15-expert-tiers.md) shows 77 % VRAM hits of an LRU of 85 slots, so the top-85
experts of a layer carry about 75 to 80 % of the reads.

- Lookahead d = 1, K = 10, per token: recall = precision = 69 % +- 5 points (Mac, 7 to 8 token verify passes there).
- Marginal control top-10: 20 to 30 % recall (cold recall by definition about 0 to 5 %).
- Chain d = 1, K = 10: 35 to 50 % recall; d = 5: 28 to 38 %; d = 20: within 3 points of the marginal.
- Rule form at theta = 0.9: precision >= 85 % but recall below 5 % (few experts pairs are that certain).
- Partner completion: +0 to 4 points at d = 1 (the router already sees the same residual, so the partner
  information is mostly redundant); fusion +2 to 6 points.
- Cross-workload loss of a chain table: 5 to 15 points, larger for the long-context set than for code vs code.
- Dense tables 48 x 20 x 65,536 x 2 bytes = 126 MB, top-16 sparse about 10 MB.

A result outside these is a finding, not noise.

## Addendum 2026-11-15: own predictors, measured the same way

The published methods are input, not a limit. Besides the table predictors above (borrowed idea: co-activation counts,
Klotski, 2609.22471; to be checked in the research note) the replay tests predictors designed here, all with the same
sets, metrics, splits and thresholds:

- **Path chain**: the context is two layers (the sets at l and l-1, or the sets at l and the lookahead-confirmed
  experts), not one: counts over (X at l-1, X' at l) -> Y at l+d, backed off to the one-layer table.
- **Whole-set conditioning**: a multi-label softmax / logistic model per (l, d) on the 256-bit indicator of the
  routed set at layer l (instead of summing single-expert conditionals), trained by gradient steps on the counting set.
- **Learned small predictor on the router state**: per (l, d) a linear map from the 256 router logits of layer l (the
  only view of the residual in the trace; the residual itself is not recorded) to the 256 scores of layer l+d,
  trained on the counting set (ridge or a few epochs of softmax cross-entropy). Parameters 65,536 per (l, d).
- **Mixing**: lookahead logits (d = 1) plus the learned or table score, weights fitted on the counting set only.

Same thresholds apply; a learned predictor must additionally show a table size and an estimated per-layer cost
(multiply-adds per token) that fit inside the layer's time, else it is a finding, not a build. Borrowed and new
predictors are labeled as such in the results. The trace job also stores the router logits (float16) for this.

## Addendum 2 2026-11-15: ideas adapted from other fields, one row each

"Who goes with whom, and who not" is solved elsewhere; each predictor below is an adaptation and is labeled with its
origin in the results (borrowed idea, own mapping). Same sets, metrics, splits and thresholds. For each: horizontal
or vertical, table size, cost per token and layer on the CPU (multiply-adds, counted from the algorithm, not
measured), recall and precision.

| Origin | Idea here | Maps to | Name in the replay |
|---|---|---|---|
| Market basket (Apriori, FP-Growth) | support, confidence P(Y given X), lift P(XY)/(P(X)P(Y)); rules above a confidence, pruning by lift < 1 | both | `chain`, `rule`, `partner`, `prune` |
| Information theory | PMI / NPMI of a pair, score = sum of the PMI over the known experts (naive Bayes in log form) | both | `pmi`, `npmi` |
| Item-item collaborative filtering | similarity of experts by the cosine of their co-occurrence vectors over tokens; predict the neighbors of the routed experts | both | `cosine` |
| Memory-based (user-user) CF, temporal streaming (STMS, Domino) | find the k most similar training tokens by the routed sets of the last J layers, vote with what followed | vertical | `knn` |
| Markov / correlation prefetchers | context = the top-1 expert (the "miss address") of layer l, successors = the next layers' sets | vertical | `markov1` (table of top-N successors) |
| PPM, variable-order Markov chain, n-gram backoff, TAGE (several history lengths, longest confident match) | contexts of 1 to 3 layers of top-1/top-2 experts, interpolated by the support of the context (Witten-Bell style), backoff to the shorter one | vertical | `ppm` |
| Perceptron branch predictor | one linear unit per target expert over the routed-set history of J layers (batch ridge instead of online perceptron updates; same function class) | vertical | `perceptron` |
| Small Bayesian network (optional) | Chow-Liu tree over the experts of a layer as the horizontal model, only if the pairwise ones fall short | horizontal | `bayes` (if run) |

The counting set stays the one of the plan; every learned or counted model sees only it. The cost per token is
reported next to the quality so that a predictor that wins by 1 point but costs more than the layer time is visible as
such.

## Active tooling

Replay: `tools/coactivation.py` (numpy, CPU only, `nice -n 19`, no GPU while another job holds the queue). Server
job: `TONA_ROUTE_LOG`, `TONA_ROUTE_TRACE_P`, `TONA_DECODE_DIAG=1`; no profiler, no `TONA_DETERMINISTIC` (the output
is a deterministic function of the model and the prompt either way, greedy; no speed is claimed). Conditions:
`tools/hw_report.py` and `tools/quiet_check.py` before and after the job.

# Results

## Run

Trace job (queue label `154 coactivation trace`, 2026-11-15 23:54 to 2026-11-16 00:00): build `tiers` (main 4259191),
26 GB class, MTP on, greedy, `TONA_ROUTE_LOG` + `TONA_ROUTE_TRACE_P`, `tools/coactivation_drive.py`: 33 requests, 14,115
token rows (3,090 passes, rejected drafts included): codeA 3,961 rows (3,049 decoded tokens), codeB 3,989 (3,059),
text 5,350 (3,837), long 815 (600, contexts of 19,363 and 23,257 tokens). The request cut from the route log found all
33 requests. Decode rates of the job (26 GB class, other workers on the CPU, not a speed claim): 49.6 / 51.7 / 26.7 /
27.5 tok/s for codeA / codeB / text / long. Files in `/var/tmp/claude/154/res/` (1.4 GB trace, not committed).

Test conditions (`tools/hw_report.py` before the job; the replay is CPU only, `nice -n 19`): Ryzen 5 9600X, RX 9060 XT
gfx1200 15.9 GiB (1.69 GiB held by the desktop), RAM 60.5 GiB (56.1 available), swap 2.12 GiB in use, load average
2.28 (1 min). Quiet check before: warning (8.8 % of the CPU threads used by other programs, 1.69 GiB VRAM held by
others), measured anyway (the output is a deterministic trace). After: "do not measure" (12.1 % CPU by others, load 5.1,
14.9 GiB VRAM held: my own server was still up when the check ran). The replay itself ran on a machine with other
workers; its output is a deterministic function of the trace, no timing is claimed.

Replay: `tools/coactivation.py --suite` (layers 1, 4, 7, ..., 46: every third, 16 layers, same layers for every
predictor), mixer `--only-mixer`, tables `tools/coact_report.py`; full tables in
[2026-11-15-expert-coactivation-replay.txt](2026-11-15-expert-coactivation-replay.txt), online and mixer numbers in
`-online.json`, `-mixer.json`, partners in `-partners.json`. Counting set and test set are always different requests.
All numbers are per token and layer, K = 10 predicted experts unless said, in percent. "Cold" = outside the 85 most
used experts of the layer **in the counting set** (so the cold set differs between rows with different counting sets:
cold numbers are comparable within a row, not across counting sets).

## 1. Reference: the lookahead and the controls

| Quantity | Expected | Measured (codeA -> codeB) |
|---|---|---|
| Lookahead d = 1, K = 10, recall = precision | 69 +- 5 | **70.3** (token), 73.2 recall / 74.1 precision per verify-pass union; on code -> text 71.3, text -> code 70.2, short -> long 72.0 |
| Marginal control (top-10 by frequency of the layer) | 20 to 30 | **20.2** (code), 3.2 (counted on code, tested on text), 5.3 (text -> code), 5.7 (short -> long); random is 3.9 |
| Lookahead, cold part (K = 10 over all experts) | not stated | cold recall 61.8, cold precision 53.6 |
| Lookahead, top-K among the cold experts only, K = 4 / 10 | not stated | cold recall 61.3 / 84.4 at cold precision 39.9 / 21.9 |

The router lookahead reproduces the Mac's 69 / 69 on the PC Coder and is stable across workloads (70 to 72 in every
experiment): it reads the real residual. Everything else below is a table or a model that sees only the expert ids
(or the 256 router logits) of earlier layers.

## 2. Vertical: layer l -> l + d (codeA -> codeB, K = 10, token level)

| Predictor (origin) | d=1 | d=2 | d=5 | d=10 | d=20 | table / cost per token and layer |
|---|---:|---:|---:|---:|---:|---|
| `marginal` (control) | 20.2 | 20.4 | 20.2 | 20.2 | 20.1 | 256 counts per layer |
| `chain` (borrowed: ST-MoE, Klotski; mean P(Y given X) over the 10 routed X) | 42.6 | 38.9 | 39.3 | 35.9 | 36.2 | dense 98 MB for d = 1..20 (16 bit), 2,560 MAC per d |
| `cosine` (item-item CF) | 46.4 | 41.9 | 42.3 | 38.3 | 38.6 | same table |
| `pmi`, `npmi` (information theory) | 10.8, 5.4 | 9.9 | 11.7 | 7.1 | 7.7 | same; favors rare pairs, a negative result |
| `path3` (own: chains from layers l, l-1, l-2, weights 1 / 0.5 / 0.25) | 43.2 | 41.0 | 41.0 | 36.7 | 36.6 | three tables, 7,680 MAC |
| `markov1` (correlation prefetcher: top-1 expert as context, top-10 successors) | 35.1 | 32.8 | 33.1 | 30.3 | 31.8 | 256 x 10 per (l, d), 10 lookups |
| `ppm` (variable-order Markov, TAGE-like backoff over top-1 / top-2 contexts of 1 to 3 layers) | 40.4 | 37.4 | 37.7 | 33.6 | 34.8 | context tables, about 1,000 MAC |
| `knn` (user-user CF / temporal streaming: 32 nearest counted tokens by the sets of 2 layers) | 49.5 | 45.8 | 46.4 | 40.7 | 41.2 | all counted tokens kept, about 1.9 M MAC |
| `ridge_set` (perceptron-style, whole routed set of layer l as input) | 50.5 | 45.9 | 46.0 | 41.4 | 42.0 | 256 x 256 per (l, d), 65,536 MAC |
| `perceptron2` (sets of layers l and l-1) | 55.6 | 51.4 | 51.6 | 45.3 | 45.4 | 131,072 MAC |
| **`ridge_logit` (own: the 256 router logits of layer l as input)** | **59.4** | **55.5** | **54.9** | **49.2** | **49.0** | 98 MB, 65,536 MAC per d |
| `mix` (rank fusion of chain, ridge_set, ppm) | 48.7 | 44.6 | 44.8 | 40.3 | 40.9 | |
| `rule0.9` (association rules, confidence >= 0.9, support >= 20) | recall 1.5, precision 80.2, 0.2 experts | | recall 0.5, 80.8 | 0.3 | 0.0 | |
| `rule0.7` / `rule0.5` | 12.6 / 62.0, 32.7 / 44.4 | | 8.0 / 62.1, 26.6 / 44.9 | | 4.8 / 59.1, 18.7 / 42.6 | |

Findings:

1. **The decay with distance is flat.** From d = 1 to d = 20 the best predictor loses 10 points (59.4 -> 49.0), the
   chain 6 (42.6 -> 36.2); the marginal control is 20. A d = 20 prediction is 29 points above the control, because
   most of what it knows is the workload state of the sequence, not the path of a token through the layers.
2. **The plain count chain is far below the lookahead** (42.6 against 70.3 at d = 1, against about 85 reported by
   ST-MoE on 60 to 160 experts and 59 by Klotski on 8): on 256 experts with top-10 and a code workload a table over expert ids
   alone is weak. A learned linear map from the router logits does better by 17 points than the table (59.4) and is
   still 11 below the lookahead at d = 1.
3. **What does not help**: PMI and NPMI (rare pairs dominate), the top-1 correlation prefetcher (35.1), path
   evidence of three layers (+0.6 over the chain at d = 1), the `mix` of weak predictors.
4. **Association rules are sharp and empty**: confidence 0.9 gives precision 80 % but names 0.2 experts per token
   (recall 1.5 %); the rule confidences are not calibrated across workloads (precision of rule0.9: 80 in code, 52
   code -> text, 38 text -> code).
5. **Cold part.** In the top-K among the cold experts only (what a prefetch of non-resident experts would do):

| Predictor | K | d=1 cold recall / cold precision | d=5 | d=10 | d=20 |
|---|---|---|---|---|---|
| `lookahead` | 4 | 61.3 / 39.9 | | | |
| `lookahead` | 10 | 84.4 / 21.9 | | | |
| `ridge_logit` | 4 | 44.0 / 28.6 | 39.6 / 24.0 | 34.2 / 21.7 | 32.5 / 19.5 |
| `ridge_logit` | 10 | 64.3 / 16.7 | 58.7 / 14.2 | 52.4 / 13.3 | 49.4 / 11.8 |
| `chain` | 10 | 51.2 / 13.3 | 46.6 / 11.3 | 42.0 / 10.7 | 40.8 / 9.8 |
| `marginal` | 10 | 12.8 / 3.3 | 13.4 / 3.3 | 13.3 / 3.4 | 13.4 / 3.2 |

   Each layer has about 2.5 cold experts per token (hits divided by recall). At d = 10 the best predictor names 10
   cold experts of which 1.3 are needed: 6.5 reads that are not needed per 1 that is (7.5 at d = 20), against 3.6 for the lookahead at
   K = 10 (cold precision 21.9) and 1.5 at K = 4.

## 3. Verify-pass unions (codeA -> codeB)

The union of the predictions of the 1 to 8 tokens of a pass against the union of what the pass routes (the set a
prefetch has to cover): lookahead 73.2 recall / 74.1 precision at K = 10 per token (86.2 / 57.7 at K = 16, 92.6 /
44.1 at K = 24); `ridge_logit` d = 1 59.8 / 69.6, d = 5 56.4 / 65.5, d = 20 51.6 / 61.7; `chain` d = 5 40.8 / 54.3.
The union precision is higher than the token precision for every predictor because the tokens of a pass share
experts, and the gap between the lookahead and the others stays: unions do not rescue a weak per-token predictor.

## 4. Horizontal: partners inside a layer (d = 1, codeA -> codeB, K = 10)

Structure of the layer (market-basket view, layers 2, 6, ..., 46, pairs with at least 20 expected co-occurrences,
file `-partners.json`): of the supported pairs 26.9 % have lift < 0.2 in the counting set, and in the test set those
same pairs have a mean lift of 0.12 (the next bins: 0.2-0.5 -> 0.42, 0.5-1 -> 0.78, 1-2 -> 1.33, 2-5 -> 2.65, above 5
-> 6.33): "who does not come with whom" is a real and stable structure inside a workload. Across workloads it
weakens: code -> text the first bin has 0.24, 0.2-0.5 has 0.63, 1-2 has 1.86; the best partner of an expert has
P(partner given expert) = 58 % in the counting set and 49 % in the test set (code -> code), 27 % (code -> text) and
30 % (text -> code).

| Predictor from the lookahead (all K = 10, per token) | recall = precision | cold recall of the K = 10 set | cold-only K = 10 cold recall / precision |
|---|---:|---:|---|
| `lookahead` | 70.3 | 61.8 | 84.4 / 21.9 |
| `look+prune` (drop experts with mean lift < 0.5 against the top-3) | 70.2 | 60.7 | 81.0 / 21.1 |
| `look+partner3` (top-3 seeds, fill from partners) | 50.0 | 24.7 | 53.2 / 13.8 |
| `look*partner3` (rank fusion lookahead and partner score) | 64.1 | 37.5 | 84.4 / 22.0 |
| `look+chain` (rank fusion) | 62.2 | 31.6 | 85.2 / 22.2 |
| `look+ridge_logit` (rank fusion) | 70.4 | 44.5 | 86.6 / 22.5 |
| fitted mixer, lookahead + ridge_logit (logistic regression, cross-fitted) | 70.2 | 43.1 | 86.0 / 22.4 |
| fitted mixer + chain + partner | 70.0 | 43.8 | 85.8 / 22.3 |
| fitted mixer, the same, counted on code, tested on text | 59.4 to 61.5 (lookahead 71.3) | 52.0 to 55.4 | 77.7 to 78.2 / 54 (lookahead 82.3 / 57.1) |

Completion from a few seeds is much worse than the lookahead's own ranks (the lookahead sees the residual, the partner
table only the ids); pruning by lift changes nothing (the experts it removes are already low in the lookahead's rank);
the best fusion lifts the cold-only recall by +1.6 to +2.2 points (86.0 to 86.6 against 84.4) and the all-expert recall
by +0.1; the fitted mixer, trained on code, **loses 10 to 12 points on text**.

## 5. Stability across workloads (K = 10, recall; cold recall / cold precision of the cold-only set in the compact table of the replay file)

| Counted on -> tested on | counting / test rows | lookahead | `marginal` | `chain` d=1 / 5 / 20 | `ridge_logit` d=1 / 5 / 20 | `knn` d=5 |
|---|---|---:|---:|---|---|---:|
| codeA -> codeB | 3,961 / 3,989 | 70.3 | 20.2 | 42.6 / 39.3 / 36.2 | 59.4 / 54.9 / 49.0 | 46.4 |
| code (even requests -> odd requests) | 3,672 / 4,278 | 69.5 | 18.5 | 39.3 / 37.0 / 33.5 | 57.0 / 52.3 / 46.6 | 44.5 |
| code -> text | 7,950 / 5,350 | 71.3 | 3.2 | 20.0 / 16.1 / 12.4 | 32.1 / 25.8 / 21.8 | 26.1 |
| text -> code | 5,350 / 7,950 | 70.2 | 5.3 | 21.0 / 18.0 / 14.4 | 31.6 / 26.1 / 20.4 | 23.0 |
| text (even -> odd requests) | 2,577 / 2,773 | 71.7 | 7.0 | 27.6 / 25.2 / 23.3 | 41.8 / 37.2 / 33.5 | 31.8 |
| short (code and text) -> long (19k and 23k context) | 13,300 / 815 | 72.0 | 5.7 | 33.2 / 28.4 / 24.5 | 52.3 / 44.7 / 38.8 | 37.5 |

- **A table counted on one workload loses 20 to 28 points on another** (`ridge_logit` d = 5: 54.9 in code, 25.8 on text
  from code, 26.1 on code from text); the loss is larger than the 5 to 15 points expected. The hot set itself
  moves: the 85 most used experts per layer of codeA cover 73.3 % of the uses in codeB, 36.3 % in the long requests and
  30.1 % in text (random is 33 %: the text uses the code's rarely used experts as much as its hot ones).
- Text is a harder target than code even in its own table (text within: 37.2 against 52.3 at d = 5), with 10 prompts
  of varied subject and German: the routing of text is less concentrated.
- The long-context decode (19k and 23k tokens) is predicted almost as well from short requests (44.7 at d = 5) as code
  from code: **context length is not the problem, the kind of text is.**
- The lookahead does not care (70 to 72 everywhere).

## 6. Shipped or online (chain table, K = 10, token recall at d = 1 / 5 / 10)

Streaming test (blocks of 8 passes, predicted before the update; layers 1, 4, ..., 46; `-online.json`):

| Test stream (counted on) | static table | online from the shipped table, half-life 500 | online from empty, half-life 500 | online marginal, half-life 500 (control) |
|---|---|---|---|---|
| code, odd requests (even requests) | 39.3 at d=1 (the chain of section 2) | 44.7 / 42.2 / 38.0 | 44.6 / 42.1 / 38.1 | 22.5 / 23.3 / 22.6 |
| text (code) | 20.0 / 16.1 / 14.6 | 45.8 / 41.9 / 39.7 | **47.6 / 43.9 / 42.1** | 19.2 / 19.3 / 20.1 |
| code (text) | 21.0 / 18.0 / 15.8 | 45.3 / 42.6 / 38.6 | 45.7 / 43.0 / 39.3 | 22.7 / 23.3 / 22.9 |
| long (short requests) | 33.2 / 28.4 / 25.8 | 38.3 / 33.2 / 30.1 | 44.4 / 40.5 / 38.1 | 18.7 / 19.4 / 19.3 |

Without decay the online table from empty still reaches 43.9 / 39.2 / 37.2 on text; half-lives of 500 tokens beat 5,000
by 3 points and no decay by 4 to 6. The online count chain is **+20 to +25 points over the online marginal at every
distance**, so the conditioning on the routed experts adds real information beyond recency and popularity, and it
**adapts to a shifted workload within the stream** (code table on text: 20.0 static -> 45.8 online). A shipped table
gives no start advantage: on text it is 1.8 points below the empty start at d = 1 (45.8 against 47.6), on long 6 points
below.

## 7. Tables and cost

Dense chain tables for d = 1..20 over 48 layers: 750 (l, d) pairs x 256 x 256 x 2 bytes = **98 MB** (the plan said 126
MB, counting all 960 pairs); top-8 / 16 / 32 successors per expert 3.1 / 6.1 / 12.3 MB. 46 % of the d = 1 entries are
non-zero. The within-layer partner table 6.3 MB. Cost per token and layer (counted from the algorithm, not
measured): chain, cosine, PMI 2,560 multiply-adds per distance; `ridge_logit` 65,536 per distance, all 20 distances at
once 1.3 M (63 M multiply-adds per token over 48 layers); `knn` 1.9 M per layer on 8,000
counted tokens (not viable); `ppm` and `markov1` below 1,100.

## Decision against the thresholds of the plan

- **Vertical chain, shipped static table: no build.** The plan's bar was at d >= 5 cold recall >= 50 % at precision >=
  50 % and +15 points over the marginal on the cross-workload test. Cross-workload (code -> text) the best predictor
  has 25.8 % at d = 5 (floor 35 %: no build). In the workload's own table the bar is met in the all-expert measure
  (`ridge_logit` d = 5: 54.9 against 20.2) but the plan's "precision >= 50 %" is not reachable in the cold-only
  measure: the cold precision of even the lookahead is 21.9 % at K = 10 (a cold-only set has about 2.5 hits to fish
  for), my plan did not see that; the honest reading is the reads per hit: 6.5 at d = 10 against 3.6 (K = 10) and 1.5
  (K = 4) for the lookahead.
- **Lead time through a chain at d >= 5 for SSD reads: no build on these numbers.** The classes where SSD lead time
  matters (10 to 16 GB) are bandwidth-bound by the count of reads (2026-11-15-expert-tiers.md: 66 reads per token at
  the 16 GB class, 135 MB per token at 14 tok/s is 70 % of the sequential rate); a predictor that adds 6.5 to 7.5 wasted reads
  per useful one at d >= 10 (1.5 to 2.5 at d = 1, K = 4) raises the reads several times over. At the 26 GB class (4.2 SSD reads per token) the wasted reads (tens
  per token) outnumber the useful ones by an order of magnitude.
- **Horizontal completion, pruning, fitted fusion: no build.** The bar was +5 points at equal K over the lookahead; the
  best is +0.1 (all experts) and +1.6 to +2.2 (cold-only K = 10), the fitted mixer is 0 at best and 10 to 12 points
  below on another workload.
- **Statistics: online with decay, not shipped.** The cross-workload loss (20 to 28 points) exceeds the 3-point bar for a
  shipped table by a factor of 7; an online table with a 500-token half-life from empty is as good or better than a
  shipped start in every stream and needs no file.
- **What remains interesting, not built:** an online-learned predictor (decayed counts, or recursive least squares on
  the router logits) for a mid-class case where RAM-resident but VRAM-cold experts would be fetched over PCIe (cheap
  reads, 55 GB/s) with 5 to 20 layers of lead; at 44 to 48 % recall and precision at d = 5 to 20 it is a 2x lift over
  the online marginal, but the lookahead already gives 70 % with one layer of lead, and PCIe reads of a layer
  complete within a layer. A next step would be a measured case where one layer of lead is not enough; none exists in
  the data so far.

## What is borrowed and what is new

- Borrowed idea, own measurement: cross-layer co-occurrence tables (ST-MoE 2606.15453, Klotski 2502.06888) =
  `chain`; ExFlow's conditional routing probability = the same table; next-layer router lookahead (Pre-gated MoE, Fate,
  Speculating Experts) = `lookahead`.
- Adapted from other fields: association rules and lift (`rule*`, `look+prune`), PMI, item-item CF (`cosine`),
  user-user CF and temporal streaming (`knn`), Markov / correlation prefetching (`markov1`), PPM and TAGE-like
  variable-order backoff (`ppm`), perceptron predictors (`ridge_set`, `perceptron2`), reciprocal rank fusion.
- New here: the router logits of layer l as input of a linear predictor for all layers up to 20 ahead (`ridge_logit`,
  nearest relatives PreScope and Mira, which use the hidden state and other heads); the cold-only top-K metric;
  verify-pass unions as the target; the decayed online chain with its marginal control; the cross-workload table of
  stability; the lift structure of "who does not come together".

## Open points (not measured)

- The trace has per-token top-10 and the router logits, not the residual; a learned predictor on the residual
  (PreScope-style) might reach beyond the 59.4 of the logits at d = 1 and is not excluded by these numbers.
- Rejected draft rows are in the unions (the pass really routed them); accepted-only unions were not separated.
- The replay uses every third layer; the early layers (0 to 2) and the last were not sampled.
- No end-to-end speed claim: nothing was built.
