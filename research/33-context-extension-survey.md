# 33 — Using a finished LLM beyond its training length: survey 2023-2026 and candidates for Qwen3.8-Flash-Next

Agent research (Sonnet), 2026-10-29, issue #77

Method: web search, OpenAlex title search (arXiv listing by month), arXiv abstract API, PDF download plus `pdftotext`,
Hugging Face model cards and `config.json`, GitHub issues and PRs via `gh`. Semantic Scholar citation lists were rate
limited (HTTP 429) and OpenAlex citation counts are near zero for 2026 papers, so "who cites YaRN / LongRoPE2" was
answered by topic search (about 25 queries over 2025-2026), not by a citation graph. Nothing was measured or built
here, no engine code changed. Marking per entry:
- **[read]**: PDF text or HTML read for the numbers quoted (numbers are the authors').
- **[card]**: model card or `config.json` read.
- **[issue]**: GitHub issue or PR text read.
- **[abstract]**: abstract page only. **[snippet]**: search-result snippet only (not read, treat as a lead).
- **[calc]** / **[estimate]**: own calculation / own reasoning, not measured.

Rates are named: **prefill**, **decode**, or a benchmark score (RULER = average of 13 tasks unless stated, MRCR = OpenAI
8-needle). "Native" = the length the model was trained on (here 262,144).

The YaRN specifics for our model (what YaRN changes at theta 1e7 and 64 rotated dims, static vs dynamic, memory and time
at 1M, product option) are in research/32. This note is the survey of everything else and does not repeat them.

Our case in one paragraph: 48 layers = 36 Gated DeltaNet (fixed state, no positions, causal conv) + 12 QSA layers
(indexer scores pooled blocks of 4, top-512 blocks = 2,048 cells + tail, partial RoPE 64 of 256, theta 1e7) + 1 MTP
layer, MoE. Positions exist only in 13 layers; the attention softmax always runs over about 2,051 cells whatever the
context length. Training: QSA is introduced during continued pretraining at 256K sequence length (Qwen3.8 tech report,
QSA "Training Details"); the earlier stage lengths of the final model are not stated in the text I read.
No training is possible for us: the model is 55 GB at IQ1_M, so even QLoRA does not fit a 16 GB card (LoRA is only an
option on a small proxy model whose results would not transfer).

---

## 0. Summary

1. **YaRN is not the best known training-free method, and it has a documented cost below the native length.** In the one
   controlled zero-shot comparison on a vendor-trained 256K model (Jet-Long, NeurIPS 2026, Qwen3-30B-A3B-Instruct-2507,
   RULER) [read]: native 256K: unextended 74.71, **YaRN 62.20**, DCA 75.36, Jet-Long 74.72. 512K: 67.22 / 59.84 / 65.70 /
   **71.99**. 1M: 26.82 / 57.71 / 56.90 / **64.03** (DNTK 52.99). Static YaRN buys 31 points at 1M and costs 12.5 points at
   the native length; a method that equals the base model below the native length and adapts above it wins on both ends.
   Qwen's own card says the same in words ("potentially impacting performance on shorter texts") [card].
2. **For hybrids with linear attention the position problem is only half of it.** The only hybrid evidence with numbers:
   Jet-Nemotron (softmax + linear layers) RULER at 128K falls to 8.5 / 5.7 (2B / 4B) without extension and reaches
   33.8 / 33.1 with a zero-shot position map [read]; GDN-NoPE hybrids trained at 4K keep single-needle retrieval to 64K
   but multi-key collapses at 4x (2 % at 16K) unless the attention is changed [read, Mechanics 1.1]. Linear layers have a
   limited receptive field (Stuffed Mamba, LongMamba, UPI); who limits a 1M run of our model (RoPE, indexer or GDN state)
   is **unmeasured**. Qwen's own number is the warning: 8-needle MRCR **96 / 93 at 128K / 256K, 40.5 at 512K, 26.4 at 1M**
   (QSA row of Table 3, Qwen3.8 tech report) [read], while RULER stays at 93.0 for 512K-1M. Retrieval of one needle
   survives; multi-needle does not. The extension method is judged on MRCR-like tasks, not on RULER NIAH.
3. **Sparse selection does not extrapolate by itself.** NSA-RoPE and InfLLM-v2 trained at 8K score 0-5 % at 4x in two 2026
   small-scale studies [read: HiLS table, FreeSolo table]. What extrapolates in those papers is the *position handling of
   the selection path* (NoPE or high-frequency-only RoPE in the retrieval step, landmark tokens, enforced sparsity in
   training). Our indexer rotates q and pooled keys with the same (YaRN-scaled) cos/sin as attention [code, research/32],
   so it is exposed to the same relative-distance shift. The indexer's recall at 512K-1M is the second unmeasured thing.
4. **Attention-temperature arguments (SSMax, log-n, YaRN mscale) are weaker here than in dense models.** They fight
   softmax dilution when the number of attended keys grows. Our softmax sees ~2,051 cells at any length, so dilution
   moves into the *indexer* (more candidate blocks), not into the attention softmax [estimate]. mscale = 1.1386 then
   multiplies the rotary part of q.k by 1.296 for no length-related reason: test mscale 1.0 (research/32 plan step 1 has it).
5. **Real 1M models are trained for it; the inference trick is only the last 4x.** DeepSeek-V4 trains 4K -> 16K -> 64K -> 1M
   (YaRN factor 16 over 65,536 baked into `config.json`) [read + card]; Kimi Linear 48B reports RULER 94.8 at 1M with NoPE
   in the full-attention layers [read]; MiniMax-M1, Nemotron 3, Granite 4 (NoPE) train long. Only the Qwen line
   (Qwen2.5-1M, Qwen3-2507, Qwen3-Next/3.5/3.6/3.8) ships "train to 256K, add DCA or YaRN x4 at inference". DeepSeek-V4's
   curve shows what a trained 1M looks like: MRCR 8-needle 0.59 (Pro) / 0.49 (Flash) at 1M [read, figure labels].
6. **Training-based methods with the best published evidence for hybrids exist and are out of reach**: DroPE (Olmo Hybrid
   adopts it: RULER 64K 85.0 vs 76.9 with YaRN [read]), HyPE/HypeNet (2.3B tokens distillation), SpectralShift
   (continued pretraining, +2.8 RULER avg), Ruiz and Gu state passing (500 post-training steps). Note: Qwen tested a NoPE
   variant of this very architecture and found "a substantially higher rate of endless generation after post-training"
   [read, tech report section 2.1.1]; DroPE would run into that.
7. **Engines are the bigger risk than the method.** llama.cpp #27756: Qwen3.8-27B (48 GDN + 16 attention layers) emitted
   EOS or lost retrieval at 98K-243K tokens on CUDA, CPU and ik_llama.cpp while **vLLM passed the same prompt with the
   same weights** (closed stale, no fix identified) [issue]. vLLM #37435: with YaRN passed by `--hf-overrides`, the **MTP
   draft did not receive it and acceptance was 0 % above the native length** (0.78-0.82 after the fix PR #37443) [issue].
   Both say: verify our own 262K+ path against an independent engine before blaming the model or the method.
8. **What is new in 2026** (section 6): zero-shot dynamic position maps (Jet-Long, LaMPE, RiPRA, MrRoPE), the first
   hybrid-specific position studies (HyPE Jan, Olmo Hybrid Apr, Mechanics 1.1 Oct), training-free GDN/Mamba treatments
   (UPI at ICLR 2026, NoteBank Oct: GDN RULER at 1M 1.7 % -> 70.3 % with a replay memory), the 1M-context serving papers
   (Windowed-MTP on Qwen3.5/3.6 at 1.04M), and DeepSeek-V4 / Kimi Linear as trained references.

---

## 1. Overview table

Columns: training-free (tf) y/n; tested length x training length; fits our hybrid y/n/?; engine effort (low = a parameter or
table, medium = kernel change in the 13 QSA layers, high = new state handling or training).
Numbers are the authors' on their models (Llama/Qwen dense unless stated), not ours.

### 1.1 RoPE scaling and frequency analysis

| Method | Year | Paper | tf | Tested x trained | Numbers | Fits us | Effort |
|---|---|---|---|---|---|---|---|
| Position Interpolation | 2023 | [2306.15595](https://arxiv.org/abs/2306.15595) | n (1K steps) | 32K from 2K | baseline of all below | n (superseded) | low |
| NTK-aware, Dynamic NTK | 2023 | bloc97 post, [HF doc](https://huggingface.co/docs/transformers/main/en/internal/rope_utils) | y | see Jet-Long | Qwen3-30B-A3B-2507 RULER 74.91 / 69.83 / 52.99 at 256K / 512K / 1M [read] | y | low |
| YaRN | 2023 | [2309.00071](https://arxiv.org/abs/2309.00071) | y (static) | Qwen3-Next x4, 262K | RULER 98.5 (4K), 93.5 (256K), 86.9 (512K), 80.3 (1M) [card]; at native 256K costs 12.5 pts on Qwen3-30B-A3B-2507 [read] | y (vendor recipe) | low (research/32) |
| ABF (base increase) | 2023 | [2309.16039](https://arxiv.org/abs/2309.16039) | n (CPT) | Qwen2.5-1M: base 1e4 -> 1e7 over 4K-262K | RULER 14B-1M at 128K: 32K stage 37.6 -> 262K stage 87.6 [read] | n | n/a |
| LongRoPE / LongRoPE2 | 2024 / 2025 | [2402.13753](https://arxiv.org/abs/2402.13753), [2502.20082](https://arxiv.org/abs/2502.20082) | n (10B tokens) | 128K from 8K (Llama3-8B) | RULER 128K 82.03 vs NTK 73.19, YaRN 49.39 [read]; short-context kept 97.6-98.6 % | n | high |
| CLEX, Resonance RoPE | 2023 / 2024 | [2310.16450](https://arxiv.org/abs/2310.16450), [2403.00071](https://arxiv.org/abs/2403.00071) | n | - | not read | n | high |
| Base bounds, HoPE, Fourier PE | 2024 | [2405.14591](https://arxiv.org/abs/2405.14591), [2410.21216](https://arxiv.org/abs/2410.21216), [2412.17739](https://arxiv.org/abs/2412.17739) | n | - | [abstract]; HoPE (RoPE only on fast dims, NoPE on slow) is the PE that extrapolates best inside HiLS [read] | ? | high |
| MrRoPE | 2026-01 | [2601.22181](https://arxiv.org/abs/2601.22181) | y | 8K -> 128K Llama3-8B; 32K -> 128K Qwen2.5-3B | RULER 128K 86.6 vs YaRN 79.9 (Llama); 53.2 vs 50.1 (Qwen2.5-3B) [read] | ? (no hybrid; <= 16x) | low |
| CoPE (soft-clip low-frequency RoPE) | 2026-02 | [2602.05258](https://arxiv.org/abs/2602.05258) | **n** (20B tokens trained with it) | 64K -> 256K, Llama-3-8B | HELMET recall 256K: RoPE 26.1, HardClip 36.9, CoPE 34.0 [read] | ? | high |
| Jet-Long (dynamic bifocal) | 2026-07 (NeurIPS 2026) | [2607.07740](https://arxiv.org/abs/2607.07740) | y | Qwen3-30B-A3B-2507 256K -> 512K, 1M; Qwen3 dense to 128K; Jet-Nemotron hybrid | see 0.1 and 2.2 [read] | **y** (hybrid tested) | medium |
| LaMPE, RiPRA, AdaGroPE, GALI | 2025-26 | LaMPE [ACL Findings 2026](https://doi.org/10.18653/v1/2026.findings-acl.1608), RiPRA [ACL Findings 2026](https://doi.org/10.18653/v1/2026.findings-acl.1058), AdaGroPE [ACL 2025](https://doi.org/10.18653/v1/2025.acl-long.28), GALI [EMNLP 2025](https://doi.org/10.18653/v1/2025.emnlp-main.443) | y | - | [abstract] only; all are variants of the same family as Jet-Long (length-aware position remapping) | ? | medium |
| Randomized YaRN, RoPE-perturbed self-distillation | 2026 | [2606.23687](https://arxiv.org/abs/2606.23687), [2604.14339](https://arxiv.org/abs/2604.14339) | n | 16K-128K | [abstract]; training recipes (BABILong, MRCR, LongBench v2 gains) | n | high |
| RoPE theory 2026 | 2026 | [2609.39929](https://arxiv.org/abs/2609.39929) (RoPE Profiler), [2602.10959](https://arxiv.org/abs/2602.10959) (base bounds), [2606.21249](https://arxiv.org/abs/2606.21249) | y (diagnostic) | - | [abstract]; zeroing the lowest-frequency pairs of retrieval heads: recall 1.00 -> 0.18 at 32 of 128 dims (Olmo/Qwen 7-8B) | ? | low |

### 1.2 Position remapping and chunking (training-free)

| Method | Year | Paper | tf | Tested x trained | Numbers | Fits us | Effort |
|---|---|---|---|---|---|---|---|
| Self-Extend | 2024 | [2401.01325](https://arxiv.org/abs/2401.01325) | y | 4x | RULER Qwen3-8B/4B/1.7B avg 84.71 / 80.84 / 67.86 to 128K, best baseline in Jet-Long Table 1 [read] | y | medium |
| Dual Chunk Attention | 2024 | [2402.17463](https://arxiv.org/abs/2402.17463) | y | Qwen2.5-1M: >=4x (32K -> 1M, 256K -> 1M) | passkey > 80 % at 1M from 32K training; 7B-1M RULER 84.4 at 128K, 14B-1M 92.2 [read]; Qwen3-2507 1M table uses DCA [card] | y (training-free) | medium; **vLLM removed its backend in v0.11** [issue #49296] |
| ReRoPE, ReAttention, STRING | 2023-24 | ReAttention [2407.15176](https://arxiv.org/abs/2407.15176), STRING [2410.18745](https://arxiv.org/abs/2410.18745) | y | STRING: Llama3.1-70B, +10 pts RULER/InfiniteBench within the trained length | [abstract] | ? (STRING fixes the under-trained far positions, not >native) | low-medium |
| StreamingLLM, InfLLM, LongHeads | 2023-24 | [2309.17453](https://arxiv.org/abs/2309.17453), [2402.04617](https://arxiv.org/abs/2402.04617), [2402.10685](https://arxiv.org/abs/2402.10685) | y | - | not read; InfLLM is the ancestor of InfLLM-v2 | n (we already have an indexer) | - |
| PoSE, CREAM | 2023-24 | [2309.10400](https://arxiv.org/abs/2309.10400), [2406.07138](https://arxiv.org/abs/2406.07138) | n (fine-tune) | - | not read | n | high |
| TokenSelect | 2025 | [EMNLP 2025](https://doi.org/10.18653/v1/2025.emnlp-main.1079) | y | - | [abstract]; token-level KV selection by QK voting, "length extrapolation" claimed; up to 23.84x attention speedup | n (we have an indexer) | - |
| DySCO (decode-time attention rescaling) | 2026-02 | [2602.22175](https://arxiv.org/abs/2602.22175) | y | 128K | up to 25 % relative on MRCR and LongBench v2 at 128K [abstract]; up-weights tokens found by retrieval heads | ? | medium |

### 1.3 Sparse and retrieval attention, and the indexer

| Method | Year | Paper | tf | Tested x trained | Numbers | Fits us | Effort |
|---|---|---|---|---|---|---|---|
| NSA, MoBA, InfLLM-v2, DSA | 2025 | NSA [2502.11089](https://arxiv.org/abs/2502.11089), MoBA [2502.13189](https://arxiv.org/abs/2502.13189), InfLLM-v2 [2509.24663](https://arxiv.org/abs/2509.24663), DSA in [2512.02556](https://arxiv.org/abs/2512.02556) | n | trained at 8K-128K | at 4x over 8K: NSA-RoPE RULER 24.8 -> 0.1 (FreeSolo) [read]; InfLLM-v2 1/0/0 at 32K (HiLS) [read] | - | - |
| Hierarchical sparse attention (HSA) dissection | 2025-10 | [2510.17196](https://arxiv.org/abs/2510.17196) | n (train) | 4K -> 32M tokens (RULER, BABILong) | [abstract]; needs a chunk encoder with CLS token, a bypassing residual path and enforced selection sparsity in training | n | - |
| HiLS-Attention | 2026-07 | [2607.02980](https://arxiv.org/abs/2607.02980) | n (native), partly (7B conversion 5B tokens) | 345M: 8K -> 4M; Olmo3-7B conversion | 345M RULER SK/MK/VT: 99/95/61 at 128K, 99/91/66 at 512K, 96/89/43 at 4M; full attention HoPE 0 at 128K [read] | n | - |
| FreeSolo | 2026-10 | [2610.17191](https://arxiv.org/abs/2610.17191) | n (CPT 0.4-1.6B) | 800M: 4K -> 16K RULER, S-NIAH to 256K | RULER 37.7 -> 30.4 (FS-B) vs transformer 37.5 -> 0.0, NSA 24.8 -> 0.1 [read]; RoPE only in the 512-token local window, NoPE for selection and readout | n | - |
| ASEntmax, Softplus attention | 2025 | [2506.16640](https://arxiv.org/abs/2506.16640), [2501.13428](https://arxiv.org/abs/2501.13428) | n | up to 1000x synthetic | [abstract] | n | - |
| Quest, RetrievalAttention, The Sparse Frontier | 2024-25 | [2406.10774](https://arxiv.org/abs/2406.10774), [2409.10516](https://arxiv.org/abs/2409.10516), [2504.17768](https://arxiv.org/abs/2504.17768) | y | - | not read | covered by research/15, 30 | - |

### 1.4 Linear attention, SSM, hybrids

| Method | Year | Paper | tf | Tested x trained | Numbers | Fits us | Effort |
|---|---|---|---|---|---|---|---|
| Stuffed Mamba, DeciMamba | 2024 | [2410.07145](https://arxiv.org/abs/2410.07145), [2406.14528](https://arxiv.org/abs/2406.14528) | DeciMamba y | - | [snippet/abstract]; Mamba2 loss collapses far beyond the training length, larger states worse (conflicting 2026 replication: [2603.12499](https://arxiv.org/abs/2603.12499)) | pure Mamba only | - |
| LongMamba | 2025 | [2504.16053](https://arxiv.org/abs/2504.16053) | y | - | [abstract]; global channels (long receptive field) fail beyond training; filters unimportant tokens before they enter the state | ? (Mamba channels, not GDN) | high |
| Ruiz and Gu, unexplored states | 2025 | [2507.02782](https://arxiv.org/abs/2507.02782) | n (100-500 steps) | 2K -> 128K, Mamba1/2, GLA | fails because the state distribution beyond the trained length was never seen; state passing or noise init fixes it [read abstract + intro] | ? | needs training |
| Mamba Modulation, MambaExtend | 2025 | [2509.19633](https://arxiv.org/abs/2509.19633), ICLR 2025 (ID unchecked) | y | up to 32x (2K -> 64K) | [abstract/snippet]; spectrum scaling of the A matrices | Mamba only | high |
| **UPI (Universal Position Interpolation)** | 2026 (ICLR 2026) | [OpenReview MjmORKLHUI](https://openreview.net/forum?id=MjmORKLHUI) | **y** | **hybrid Mamba-Transformer, 4K -> 64K** | PG-19 PPL, LongBench, RULER, short context kept [abstract only; PDF not retrievable]; combines Mamba cumulative decay scaling with rotary frequency scaling, a few auxiliary forward passes | **? (closest in spirit; Mamba2 not GDN)** | medium |
| SuPER-Mamba | 2026-10 | [2610.16636](https://arxiv.org/abs/2610.16636) | y | LongBench-E | avg 8.06 -> 15.69 [abstract] | Mamba only | - |
| PoST, SpectralShift | 2026 | [2604.07658](https://arxiv.org/abs/2604.07658), [2609.14320](https://arxiv.org/abs/2609.14320) | n | 8K -> 128K (SpectralShift) | RULER avg 52.38 -> 55.18 with CPT [read]; reparametrises the GDN alpha projection | n | needs training |
| CST (credit stabilisation) | 2026-09 | [2609.09157](https://arxiv.org/abs/2609.09157) | n | up to 128x [abstract] | training-time BPTT fix | n | - |
| **NoteBank** | 2026-10 | [2610.15715](https://arxiv.org/abs/2610.15715) | **y** | 4K-trained GDN/Mamba backbones to 1M; Qwen3.5-9B with YaRN | GDN 1.3B mean of 6 RULER retrieval tasks at 1M: **1.7 % -> 70.3 %**; Qwen3.5-9B LongBench v2 46.1 -> 49.3, +10-13 % latency [read] | ? (replay memory in the harness layer, not in the kernel) | high |
| HyPE / HypeNet | 2026-01 | [2601.22156](https://arxiv.org/abs/2601.22156) | n (2.3B tokens distillation) | Qwen3 -> hybrid; 93.5 % NIAH at 64x training length | RoPE in the RNN layers, NoPE in attention, dynamic logit scaling s_t = log_a(t + a) at inference [read] | n | - |
| Olmo Hybrid (DroPE vs YaRN) | 2026-04 | [2604.03444](https://arxiv.org/abs/2604.03444) | n (100B tokens CPT) | 7B GDN hybrid, RULER 4K-64K | YaRN 92.8/91.3/90.0/84.7/76.9, DroPE 92.2/89.8/88.4/86.2/85.0 [read] | n (training) but best hybrid evidence | - |
| **Mechanics of Long-Context Hybrid Models 1.1** | 2026-10 | [2610.10114](https://arxiv.org/abs/2610.10114) | **y** (EME) | 776M hybrids 4K -> 64K | GDN-NoPE-LH NIAH-SK2 at 16K: 2 % -> 99 % with EME [read]; see 2.4 | ? (needs NoPE attention layers; ours is RoPE) | high |
| Jet-Nemotron + Jet-Long | 2026-07 | [2607.07740](https://arxiv.org/abs/2607.07740) | y | 2B/4B hybrid, native 32K -> 128K | RULER 128K 8.54 -> 33.78 (2B), 5.65 -> 33.14 (4B) [read] | y (hybrid, softmax layers use RoPE) | medium |
| Samba (SWA + Mamba) | 2024 | [2406.07522](https://arxiv.org/abs/2406.07522) | y (architecture) | 4K -> 1M PPL, passkey 256K after 500 steps | [read abstract + intro] | n | - |
| Kimi Linear (KDA + NoPE MLA) | 2025-10 | [2510.26692](https://arxiv.org/abs/2510.26692) | n (trained) | RULER 94.8 at 1M (5.7T-token version) | NoPE vs RoPE: better long-context scores, same short [read] | n | - |

### 1.5 Attention temperature and logit scaling

| Method | Year | Paper | tf | Notes | Fits us |
|---|---|---|---|---|---|
| SSMax (Scalable-Softmax), logits x s log n | 2025-01 | [2501.19399](https://arxiv.org/abs/2501.19399) | partly (can be inserted after pretraining, gains partial) | [abstract] | weak (our softmax size is constant) |
| Critical attention scaling | 2025-10 (ICLR 2026) | [2510.05554](https://arxiv.org/abs/2510.05554), [2605.12697](https://arxiv.org/abs/2605.12697) | theory | beta_n of order log n is critical; replica analysis gives sqrt(log n); the YaRN factor appears as (log n)^2 in their table [snippet] | n/a |
| SWAN-GPT dynamic scaling, HyPE scaling | 2025-04 / 2026-01 | [2504.08719](https://arxiv.org/abs/2504.08719), [2601.22156](https://arxiv.org/abs/2601.22156) | y (inference) | logits scaled by log_a(t + a), a fitted after training on pretraining documents [read for HyPE] | for NoPE layers only |
| Llama 4 temperature tuning | 2025 | vLLM `llama4.py` | y | applied to NoPE layers only, `floor_scale` 8192, `attn_scale` 0.1, recommended above 32K [snippet] | n/a |
| YaRN mscale | 2023 | [2309.00071](https://arxiv.org/abs/2309.00071) | y | 0.1 ln s + 1 on cos and sin | see 2.5 |
| DySCO | 2026-02 | [2602.22175](https://arxiv.org/abs/2602.22175) | y | see 1.2 | ? |

---

## 2. Details per category

### 2.1 RoPE scaling and frequency analysis

- **Where the damage comes from.** LongRoPE2's hypothesis (higher RoPE dimensions are under-trained, real data has few
  long dependencies, so the empirical critical dimension sits earlier than theory says) is supported by one ablation:
  correcting only the critical dimension lifts YaRN at 128K from 49.39 to 71.46 and NTK from 73.19 to 77.25 on Llama3-8B,
  but the full method still wins at 82.03 [read]. Search for factors needs the model and a needle PPL objective
  (10 PG19 books, a needle inserted at the start, PPL on the needle tokens only: RULER 128K 82.03 vs 78.68 for plain
  PPL search) and then **mid-training with the factors** (10B tokens; they list the one-time KV recomputation when switching
  RoPE as a limitation). No training-free LongRoPE2 result exists; "search only" was not evaluated [read].
- **Perplexity misleads.** Section 4 below. For extension methods specifically: PG-19 PPL puts Jet-Long, Self-Extend and
  DCA within 0.1 PPL of each other (10.10 / 10.20 / 10.14) while RULER separates them by up to 5 points (81.2 / 77.8 / 76.4).
  Static YaRN (Qwen3 base models, up to 128K from 32K) is *worse than doing nothing* on RULER at 1.7B (52.99 vs 60.93, PPL
  16.39 vs 16.13 also worse), equal at 4B (70.11 vs 69.94) and better only at 8B (78.49 vs 73.13), whereas DNTK, DCA,
  Self-Extend and Jet-Long gain 8-13 points at every size [read, Jet-Long Table 1].
- **Low-frequency handling is where 2026 papers look.** CoPE soft-clips the lowest frequencies (trained model, 64K -> 256K,
  HELMET Summ 32.4 vs 9.1 for RoPE at 256K; recall 34.0 vs 26.1) [read]; DaRoPE replaces absolute position on the slow
  bands by data-dependent bounded coordinates (124M-50B, trained) [abstract]; HoPE/HiLS use RoPE only on dimensions whose
  period fits the training length [read]. For us (research/32 section 1.2): pairs 22-31 (10 of 32) have wavelength > 262K
  and are the ones YaRN divides by 4. [calc, own, same table as research/32]: every one of these pairs sees 4x its trained
  maximum angle at 1M. In absolute terms pairs 22-27 leave the trained range by more than 0.9 rad (pair 24: 5.9 rad at 1M vs
  1.47 rad at 262K; pair 27: 1.3 vs 0.33), pairs 28-31 stay below 0.8 rad. So the strongly shifted part is about pairs 22-27
  (6 of 32), the rest is a small extrapolation of a nearly constant phase.
- **What the vendors do with RoPE when they train for long**: Qwen2.5-1M raises the base 1e4 -> 1e6 (32K) -> 5e6 (128K) -> 1e7
  (256K) [read]; Qwen3-Next/3.5/3.8 use 1e7 and extend by YaRN x4 [card]; DeepSeek-V3.2 uses YaRN factor 40 over a 4,096 base
  *during training* (163,840 positions), V4 factor 16 over 65,536 (1,048,576) [card, config.json]; Kimi-K2-0905 factor 64 over
  4,096 (262,144); gpt-oss factor 32 over 4,096 (131,072, `truncate: false`, llama.cpp PR #30206 open) [card]; Falcon-H1 uses
  theta = 1e11, i.e. near-NoPE, at 262,144 [card]; Granite 4 and Kimi Linear MLA use no RoPE [card/read]. In every one of
  these YaRN is part of the training run; "static YaRN at inference on a model that was not trained with it" is the Qwen
  route and the one in the table of 0.1.

### 2.2 Position remapping and chunking

- **Self-Extend / Jet-Long / DCA are the same idea in three steps**: keep exact RoPE inside a local window, map far
  positions onto the trained grid (floor division by G, or chunk-wise repeated positions). Jet-Long sets G = ceil(L / native)
  (G = 2 at 512K, 4 at 1M for us), keeps w0 = 2,048 local tokens exact, stores keys at their true positions and applies a
  correction rotation to the query and to each *gathered* key (additive angle composition, so no rewrite of the cache)
  [read, section 3]. Within the native length it is bit-identical to the base model (f(x) = x).
- **Results on a 256K-native model, zero-shot** (Qwen3-30B-A3B-Instruct-2507, RULER, three seeds, 100 examples per task) [read]:

  | | 256K (native) | 512K (2x) | 1M (4x) |
  |---|---|---|---|
  | unextended | 74.71 | 67.22 | 26.82 |
  | YaRN | 62.20 | 59.84 | 57.71 |
  | DNTK | 74.91 | 69.83 | 52.99 |
  | DCA | 75.36 | 65.70 | 56.90 |
  | Jet-Long | 74.72 | 71.99 | 64.03 |

  Self-Extend is omitted there (two rotated key views are too costly in a standard implementation). Local window ablation:
  w0 = 0 collapses RULER to 1-5; any w0 in 512-4,096 is within 2 points of the best (Qwen3-4B/8B, 64K/128K) [read].
  Position aliasing beat continuous frequency interpolation in their ablation (Table 6, numbers not copied).
  Caveats: dense softmax model, one family, no multi-needle/MRCR number, 4x at most; the unextended model at 512K (67.22)
  is already decent, so on a 256K-trained model 2x is nearly free and the first real test is 4x.
- **Why this matters for a sparse-gather engine** [estimate]: our decode gathers 2,051 cells per layer for one query. A
  correction rotation per gathered key is 2,051 x 2 KV heads x 32 pairs per layer, negligible next to the host-RAM gather
  (research/32 option C is the same mechanism). Prefill is harder: every query has its own 2,048 cells, so the
  correction is per (query, key) pair; Jet-Long uses three FlashAttention calls merged by log-sum-exp, which does not map
  to our QSA prefill kernel. Order of magnitude of the extra FLOPs for per-pair rotation: ~4 % of the QK work (1M flop vs
  25M per token and layer) [estimate]. The pooled indexer keys are stored rotated at the block's first position; a
  second (remote) view of 262K pool entries per step is too expensive, so the indexer would keep native RoPE or get a
  one-time re-rotation per G change (as in research/32 option B).
- **DCA in the wild**: Qwen2.5-1M's own numbers (DCA + YaRN attention scaling, "no change below the trained length"):
  14B-1M RULER 95.7 avg, 92.2 at 128K (Table 4), passkey over 80 % at 1M from a 32K-trained model (Figure 3) [read].
  The MInference-style sparse prefill they stack on top showed a drop with DCA ("non-continuity of relative positions may
  disrupt the slash pattern") and needed a refinement [read]: a warning for any remap combined with sparse selection.
  Qwen3-2507 cards use DCA for their 1M RULER table (Qwen3-30B-A3B-2507 72.8 at 1M, 235B 84.5) [card]; vLLM removed the
  DCA backend with the V0 engine in v0.11 and nobody ported it [issue #49296].
- **Not needed for us**: StreamingLLM, InfLLM, LongHeads, TokenSelect do selection in the attention; QSA's indexer does that.

### 2.3 Sparse and retrieval attention as extrapolation

- **No free lunch**: models with a sparse selection layer trained at 4K-8K fail past that length exactly as dense RoPE
  does when the selection or the readout uses RoPE: NSA-RoPE at 4x: RULER mean 24.8 -> 0.1 (FreeSolo, 800M, trained 4K, tested
  16K) [read]; HiLS table (345M, trained 8K): NSA-RoPE 49/17/16 at 8K and 5/0/0 at 32K; InfLLM-v2 77/39/25 and 1/0/0;
  full attention with HoPE 72 at 32K and 0 at 128K [read].
- **What extrapolates** is a design with (a) NoPE or fast-dims-only positions in the selection/readout path, (b) landmark
  tokens or compressed block keys trained for retrieval, (c) enforced sparsity during training so test and training
  distributions match [HSA 2510.17196 abstract; HiLS and FreeSolo read]. HiLS 345M: RULER 99/95/61 at 128K and
  99/91/66 at 512K trained at 8K (64x), and a 7B Olmo3 conversion with landmark-token tuning (<1 % of parameters, 5B
  tokens) lifts RULER 128K from 0.00 to 94.67 [read]. Our QSA fits (b) and (c) (Qwen trains the indexer by KL against dense
  attention, then sparse training, tech report QSA "Training Details") but not (a): it is RoPE'd in both indexer and attention.
- **QSA vs dense at long context** (Qwen3.8 tech report Table 3, same backbone, CPT at 256K) [read]:

  | | RULER <=128K | 128-256K | 256-512K | 512K-1M | MRCR 8n 128K | 256K | 512K | 1M |
  |---|---|---|---|---|---|---|---|---|
  | full attention | 99.84 | 99.81 | 97.65 | 90.08 | 97.14 | 94.20 | 30.66 | 20.71 |
  | QSA | 99.89 | 99.62 | 98.95 | 93.00 | 95.98 | 93.00 | 40.53 | 26.44 |

  Reading: sparse selection is not what breaks beyond 256K (it is better than dense there); the break is between
  256K and 512K for multi-needle tasks in *both* variants. The tech report does not say which RoPE extension was used
  for lengths above 262,144 (no YaRN mention in the text; the card says YaRN x4) [read]. DSA at 128K training, V4 at 1M training:
  V4-Pro MRCR 8-needle ~0.59 and V4-Flash ~0.49 at 1M after 0.82/0.76 at 256K (Figure 9 labels) [read], i.e. a *trained* 1M
  model also loses ~25 points between 256K and 1M on this task. Top-k of V4: 1024 (Pro) vs 512 (Flash) [card].
- **The indexer itself is the untested part at length.** The 2,048-cell budget is constant while candidate blocks grow 4x
  from 256K to 1M; the true top blocks have more competitors, the score noise per block is unchanged. No paper measures
  indexer recall against dense attention mass as a function of length beyond training; IndexCache, Kascade and
  LongCat/StreamIndex (research/29, 30) are about speed. A cheap offline probe exists: dense scores for a few queries
  at 512K are affordable with the CPU reference or a chunked dense kernel [estimate].

### 2.4 Hybrids with linear attention or SSM

- **Generic statement**: Mamba-type models trained at 2K-8K lose language-modelling quality far beyond the trained length;
  causes proposed: state too large to forget (Stuffed Mamba), transition spectrum / decay (DeciMamba, Mamba Modulation),
  unexplored state distribution (Ruiz and Gu: "models fail to length generalize when during training they are only exposed to
  a limited subset of the distribution of all attainable states"; effective remembrance concentrates on the first tokens) [read].
  A 2026 image-reconstruction study could not reproduce the overparameterisation claim (2603.12499) [snippet].
  Gated DeltaNet is better than Mamba2 at extrapolation in its own paper (NIAH degrades later) [snippet on 2412.06464], but the
  tested scale is 1.3B.
- **Only mixed evidence on GDN hybrids at scale beyond training**: Qwen3-Next RULER 80.3 at 1M with YaRN x4 (vs 93.5 at
  256K, 98.5 at 4K) [card]; Qwen3.8 MRCR cliff (2.3); community reports at 1M on Qwen3.5/3.6/3.8 with YaRN factor 2-4 working in
  vLLM and SGLang, e.g. vLLM #62776: Qwen3.8-Flash-Next with YaRN factor 2 at 524K retrieves correctly up to 181,635 prompt tokens on
  one GB10 (a TP=2 variant of the same report failed for unrelated, not understood reasons) [issue]. The 1M serving paper
  Windowed-MTP runs Qwen3.6-35B and Qwen3.5-122B (GDN + attention) at a 1,040,000-token prompt with YaRN and states "positional validity
  at 1M is handled at the model level" [read], but publishes no accuracy number of the target.
- **Hybrid-specific position theory (2026)**:
  - *HyPE* (Tsinghua): RNN layers carry position (RoPE), attention layers use NoPE; "length generalization depends only on
    the attention layers" because RNN layers are agnostic once the context exceeds their receptive field; dynamic logit
    scaling log_a(t+a) at inference [read].
  - *Olmo Hybrid* (AI2): GDN hybrid 7B, DroPE beats YaRN at 64K (85.0 vs 76.9) and they ship DroPE; their explanation "the
    GDN layers carry implicit positional information through their recurrent structure, so the attention layers are less
    dependent on explicit positional encodings" [read].
  - *Mechanics 1.1*: trade-off, GDN hybrids gain more from continued training, SWA hybrids extrapolate better zero-shot.
    EME = sliding window on the position-biased layers (SWA for RoPE, "SWLA" = windowed chunks for GDN) + log scaling on NoPE
    layers: 776M GDN-NoPE-LH NIAH-SK1 100 % from 4K to 64K, SK2 90 %, SK3 53 %; baselines SK2 2 % at 16K [read]. Not applicable as is:
    needs NoPE attention layers, and a window on GDN removes exactly what GDN is for.
  - *Kimi Linear*: NoPE in the full layers "facilitates straightforward context window extension", RoPE variant worse on long context
    [read]; same logic as HyPE.
- **Training-free treatments of the linear part** (all for Mamba, none validated on GDN at scale): LongMamba (filters
  unimportant tokens in global channels), Mamba Modulation (spectrum scaling of A), MambaExtend, SuPER-Mamba, UPI (hybrid,
  closed form, ICLR 2026, "needs only a few auxiliary forward passes", 4K -> 64K) [abstracts]. The GDN analogues would
  scale the decay gate; SpectralShift shows the right quantity (slow spectral band aligned with the target dependency length,
  keep fast-decaying modes) but trains for it. **I found no training-free recipe tested on a Qwen-style GDN hybrid.**
- **A memory instead of a better state**: NoteBank archives selected text before the question is known, retrieves by the
  backbone's own features and replays it (training-free): plain GDN 1.3B mean over six RULER retrieval tasks at 1M: 1.7 %
  -> 70.3 %; Mamba-2 up to +37.8 points; Qwen3.5-9B (with YaRN) LongBench v2 46.1 -> 49.3 (+3.2), MemoryAgentBench +1.1; latency
  +10.6 to +13.2 % [read]. It addresses the state, not the RoPE, and works at harness level; not an engine feature yet.
- **Why our case may be easier than the papers**: 12 softmax-like layers with top-2,048 over the whole context see
  distant tokens directly, so the GDN layers have the job of recency/local mixing, as in HyPE's argument. The cost is that the
  *indexer and the RoPE of those 12 layers* carry the length generalisation. This is a hypothesis, to be tested by the
  distance experiment in section 5.

### 2.5 Attention temperature and logit scaling

- SSMax multiplies logits by s log n (s learned): inserting it after pretraining recovers part of the gain [abstract].
  Critical-scaling theory (ICLR 2026) supports beta_n of order log n [snippet]; a May 2026 paper says the right law depends on
  the score distribution [snippet]. YaRN's mscale is a log-n correction in the same family. Llama 4 uses temperature tuning
  only on its NoPE layers [snippet]; HyPE does so too [read].
- Our attention normalises over a fixed ~2,051 cells, so the dilution term of these laws does not grow with context
  [estimate]. The indexer has a different problem (ranking among more blocks); a log-n factor on indexer scores would not change the
  ranking. What the YaRN mscale does here is a multiplicative 1.296 on the rotary part of q.k (research/32 section 1.1) which
  changes the *balance of the 64 rotary dims against the 192 content dims* in attention and the 64 rotary against 64
  content dims in the indexer score (the indexer top-k can therefore change with mscale even at short context). Candidate
  measurement: top-k overlap of the indexer with and without YaRN at 32K-250K, by layer.
- Decode-time rescaling guided by retrieval heads (DySCO, +25 % relative on MRCR/LongBench v2 at 128K, abstract) is the only
  2026 temperature-style method with gains on the kind of task where Qwen3.8 is weakest; it needs attention weights of selected heads.

### 2.6 What the real 1M models do

| Model | Context | How (primary source) | Class |
|---|---|---|---|
| Qwen2.5-1M | 1M | staged 4K -> 262K training with ABF 1e4 -> 1e7, then DCA + YaRN scaling x4 + sparse prefill at inference [read] | train + inference trick |
| Qwen3 / Qwen3-2507 | 128K / 1M | ABF 1e6 at 32K training, YaRN + DCA "four-fold at inference" [read]; 2507 1M variants with DCA + MInference, 1M config file [card snippet] | train + trick |
| Qwen3-Coder-480B | 1M | YaRN (Unsloth 1M GGUFs) [snippet] | trick |
| Qwen3-Next / Qwen3.5 / 3.6 / 3.8-Flash-Next | 262K native, ~1M | YaRN x4 factor, "validated up to 1M" (Next), static YaRN warning [card]; hosted Flash versions "1M by default" [card] | trick (4x) |
| DeepSeek-V3.2 | 128K | DSA trained 128K (2.1B indexer warm-up + 943.7B sparse tokens), YaRN factor 40 over 4,096 in training [read + config] | trained |
| DeepSeek-V4 / V4.1 | 1M | stages 4K, 16K, 64K, 1M (sparse attention introduced at 64K), YaRN 16 over 65,536 in `config.json`, compressed-KV theta 160,000, partial RoPE 64 dims, MRCR above [read + config] | trained |
| Kimi K2-0905 / K2.5 | 256K | YaRN factor 64 over 4,096 [config]; K3 listing 1,049K unverified [snippet] | trained |
| Kimi Linear 48B-A3B | 1M | KDA : MLA 3:1, **NoPE MLA**, `rope_scaling: None`, RULER 94.8 at 1M, 84.3 avg to 128K [read] | trained |
| MiniMax-M1 | 1M | lightning : softmax 7:1, 7.5T-token CPT [snippet]; M2 reverted to full attention because of "performance degraded noticeably as context length grew" [snippet] | trained / abandoned |
| Llama 4 Scout | 10M claimed | iRoPE (chunked-local RoPE + global NoPE) + temperature tuning; trained 256K [snippet, no primary source found] | trained |
| Gemma 3 | 128K | 5:1 local(1024):global, global RoPE base 10K -> 1M, PI-like factor 8 for the extension [snippet of 2503.19786, report not read] | trained |
| gpt-oss-120b | 128K | YaRN factor 32 over 4,096, theta 150,000, alternating dense/SWA(128) [config] | trained |
| Granite 4.0 | 128K validated | Mamba-2 hybrid, NoPE, trained to 512K per a marketplace listing [snippet] | trained |
| Nemotron 3 Super/Nano | up to 1M | Mamba-Transformer MoE; NoPE per third-party only [snippet]; config has `max_position_embeddings` 262,144 and `rope_theta` 10,000 [card]; RULER at 1M claims by NVIDIA [snippet] | trained |
| Falcon-H1 | 256K | theta = 1e11, near-NoPE, "hybrids are sensitive to the RoPE base" (cited in Kimi Linear) [card + read] | trained |
| MiMo-V2-Flash | 256K | SWA(128) hybrid, theta 5e6 global and 1e4 local [config] | trained |
| GLM-4.6 | 200K | theta 1e6, no scaling [config] | trained |

Official recommendations to *users* exist for exactly one family: the Qwen YaRN x4 note above. Everyone else ships a model
trained for its length.

---

## 3. Engine and serving evidence (issues and PRs read)

| Item | What it says | Relevance |
|---|---|---|
| llama.cpp [#27756](https://github.com/ggml-org/llama.cpp/issues/27756) (closed stale 2026-10-23) | Qwen3.8-27B (64 layers, 48 GDN): silent EOS at ~100K-243K prompt tokens on CUDA, CPU, ik_llama.cpp; non-monotonic "holes" (96,519 fails, 95,979 and 99,003 pass); Qwen3.5-35B-A3B (30 GDN) passes at 243K; **vLLM 0.28 passes 129,864 tokens with the same weights** | an independent engine is the control for any >262K quality number; keep a vLLM/SGLang reference run |
| llama.cpp [#27090](https://github.com/ggml-org/llama.cpp/issues/27090) (open) | Qwen3.8-27B + YaRN x4 dies silently at ~520K prefill tokens (just under 2x native) on CUDA; YaRN x2 at 444K works with 6/6 needles; a first cliff at 90-100K was fixed in b10434 (#26623 recurrent-state rollback) | position-index overflow candidates at 2x native; check our int/float ranges at 524,288 |
| llama.cpp PR [#31371](https://github.com/ggml-org/llama.cpp/pull/31371) (closed), [#30206](https://github.com/ggml-org/llama.cpp/pull/30206) (open) | server silently caps slot context to `n_ctx_train` when RoPE scaling is set; `truncate=false` for gpt-oss-style YaRN | product detail |
| vLLM [#37435](https://github.com/vllm-project/vllm/issues/37435) / PR #37443 | draft/MTP config dropped `--hf-overrides`; Qwen3.5-4B at 265K: acceptance 0/19,998 and 0/6,870 unpatched, 0.82 / 0.78 patched; same for Qwen3.6 above 256K | **our MTP layer must use the same rope parameters as the target; test acceptance length by context length** |
| vLLM [#49296](https://github.com/vllm-project/vllm/issues/49296) | DCA backend removed in v0.11 with V0, never ported | DCA only as own implementation |
| vLLM [#62776](https://github.com/vllm-project/vllm/issues/62776) (open) | Qwen3.8-Flash-Next at TP=2 on GB10: content beyond `indexer_budget` (2,048) tokens invisible; TP=1 v0.31/v0.32 correct (retrieval at 181K with YaRN x2, 524,288 max length); not resolved | confirms that QSA correctness at depth is engine-specific; exact top-k (`VLLM_QSA_DET_TOPK=1`) used by working setups |
| SGLang [#36807](https://github.com/sgl-project/sglang/issues/36807) (open) | `fast_topk_v2` silently wrong when the radix threshold bucket exceeds a 4,096-entry candidate buffer (k=2,048; 6-42 of 2,048 wrong per row at 64 x 256K, 98-157 at 64 x 1M); flashinfer top-k correct | our radix `op_topk` (#74): check bucket overflow at 262K-1M rows; test recall against a sort |
| vLLM/SGLang model cards for Qwen3.5/3.6/3.8 | all use static YaRN via `rope_parameters` override | what everyone else runs |

---

## 4. Evaluation: what shows real length generalisation

- **Perplexity is a poor judge** (LongPPL [2410.23771](https://arxiv.org/abs/2410.23771): averaging hides the few key tokens;
  Pearson -0.96 for LongPPL vs benchmarks) [abstract]; Jet-Long above: Jet-Long, Self-Extend and DCA are indistinguishable by PG-19 PPL, not by RULER [read]. The
  controlled study 2409.12181 reaffirms PPL as indicator for *within* the extension range and says extrapolation beyond it
  remains hard and approximate attention systematically underperforms [abstract]. For us, mean dNLL (the repo's rule) is a
  necessary regression gate and not evidence of retrieval: use LongPPL-style key-token NLL (long-short contrast) and the worst window.
- **RULER** ([2404.06654](https://arxiv.org/abs/2404.06654), 13 tasks) saturates for retrieval-trained models: Qwen3.8 99.8 up to 256K, 93.0 above 512K; but multi-key
  and aggregation tasks fail first in other models (research/30 4.2). Useful as sensitivity control, not as the curve.
- **MRCR 8-needle** ([OpenAI dataset](https://huggingface.co/datasets/openai/mrcr)) separates: Qwen3.8 96 -> 26.4, V4-Pro 0.9 -> 0.59. It is the benchmark
  on which our extension must be shown to work; mean of ordered-needle similarity at 128K/256K/512K/1M.
- **HELMET** ([2410.02694](https://arxiv.org/abs/2410.02694)): NIAH does not predict downstream tasks; seven application categories, RAG score is the best single predictor
  (Jet-Long uses HELMET-RAG) [abstract]. Up to 128K only.
- **NoLiMa** ([2502.05167](https://arxiv.org/abs/2502.05167)): no lexical overlap; at 32K 11 of 13 models fall below 50 % of their short baseline (GPT-4o 99.3 -> 69.7) [abstract]. Sensitive to
  exactly the semantic-retrieval ability that a RoPE-shifted indexer may lose.
- **LongBench v2** ([2412.15204](https://arxiv.org/abs/2412.15204)), **BABILong** ([2406.10149](https://arxiv.org/abs/2406.10149), splits to 10M; popular LLMs use 10-20 % of the context), **Michelangelo**
  ([2409.12640](https://arxiv.org/abs/2409.12640), latent structure over code and text), **LongCodeBench** ([2505.07897](https://arxiv.org/abs/2505.07897), up to 1M, code repair; Claude 3.5 Sonnet 29 % -> 3 %, Qwen2.5
  70.2 -> 40 %) [abstracts]. LongCodeBench and code QA fit our priority (code first). Fiction.LiveBench and Chroma "Context Rot" were
  not read.
- **Position-bias controls matter**: the distance of the needle to the *end* and to the *start* are different variables; "RoPE Profiler"
  (2609.39929) and "Shuffle the Context" (2604.14339) report strong dependence of accuracy on the absolute placement of the
  evidence [abstract]. Always report depth *and* distance.
- **Where perplexity-good methods fail**: YaRN (above), SSMax-style temperature fixes, and anything judged on the first 1-2K
  needle positions; HiLS shows NSA-RoPE keeping PPL 19 at 512K while RULER is 0 [read].

---

## 5. Candidates for us (ranked), with test plans

All use the repo's rules: bit-exact when off, `IE_DETERMINISTIC=1`, measurements as files under `research/measurements/`,
dNLL gate +0.01 nats (research/31) and a retrieval/code gate. Lengths: **250K** (inside native, regression and cost of the
method), **512K** (2x, factor/G = 2), **1M** (4x). Build the corpus from our own repositories (code first) with
`retr.py`-style multi-key tasks, plus MRCR 8-needle and one LongCodeBench-style repair set. Budget per point: ~19 min of prefill
at 1M (research/32 section 5), so use few, long runs with several questions per context.

### C1. Attribution first: distance, depth, and the RoPE/YaRN grid (training-free, effort low; extends research/32 section 6)

Why: three possible limiters are unmeasured (RoPE OOD in 12 layers, indexer ranking, GDN state), the MRCR cliff between 256K and
512K exists in both Qwen variants, and every candidate below is a bet on one of them.
- Experiment 1, *distance vs length*: a 1M context where the same 8 needles sit (a) in the first 100K, (b) in the middle,
  (c) in the last 200K. If (c) is fine and (a) fails, relative distance beyond 262K (RoPE of the 12 layers) is the limiter; if
  (c) also fails, the indexer or the GDN state is. Same at 512K. No YaRN, YaRN x2, x4, and mscale 1.0 vs 1.1386.
- Experiment 2, *indexer recall*: for 16 queries per length at 128K, 250K, 512K, compute dense attention mass per block (reference
  path or chunked dense kernel) and measure the fraction of mass inside the selected 512 blocks and the overlap with the oracle top-512.
  Report by layer. Flat recall means the indexer is not the limiter; falling recall motivates C3.
- Experiment 3, *top-k overlap with and without YaRN at 32K-250K* per layer (does static YaRN change the indexer's choices at
  short length?).
- Gate/decision: MRCR 8-needle and key-token NLL at 250K / 512K / 1M; if mscale 1.0 matches 1.1386 above native and equals base below, ship "mscale 1.0 / factor by request length".

### C2. Zero-shot dynamic position map (Jet-Long family: local window exact, far positions floor(x/G)) (training-free, effort medium)

Why: the only zero-shot method with a hybrid result and a documented advantage over YaRN above the native length (+12 at 512K and
+6 at 1M on a 256K model), identical to base below 262K (no static-YaRN tax, which is 12.5 points on the same model).
- Implementation sketch (decode first): keys stay stored as today (rotated at true position); per gathered cell apply the
  correction rotation (delta angle = (floor(k/G) - k) * f_i on the 32 pairs, plus the same for the query), local window w0 = 2,048
  exact; G = ceil(L / 262,144) (2 at 512K, 4 at 1M). Indexer: keep native RoPE first, then try the one-time re-rotation of the pool per
  G change (research/32 option B); compare.
- Test: native 250K bit-identical (sanity); 512K and 1M against YaRN x2 / x4 (from C1) on MRCR, multi-key retrieval, NoLiMa-style
  and one code-repair set; also decode-time MTP acceptance length. Then prefill: either a per-(query, key) correction in the QSA prefill
  kernel (~4 % extra FLOPs [estimate]) or restrict the remap to decode and measure how much of the gain remains when the prompt was
  prefilled under native positions (Jet-Long Appendix E measures the same approximation: retained cache vs re-prefill).
- Kill criterion: no gain over the best YaRN setting at 1M on MRCR/multi-key, or any regression below 262K.

### C3. Length-dependent selection budget (top-k blocks 512 -> 768-1024 above 262K) (training-free, effort low-medium)

Why: C1 experiment 2 will say whether recall drops with length; DeepSeek-V4 itself uses 1024 for Pro and 512 for Flash [config]; PCIe is
not the limit (the gather is 28 MB per token; 2x budget = 57 MB, research/32 section 4). The model was trained for 512; extra cells at inference are a
mild distribution shift, so measure.
- Test: budget 512 / 768 / 1024 blocks at 512K and 1M, keep the 3 tail cells; compare MRCR, multi-key, and decode pass time
  (indexer unchanged, gather and attention up to 2x). Check that the radix top-k (#74) is correct for k = 1024 and rows of 262K
  scores (SGLang #36807 is the example of a silent failure).
- Combines with C2 and with IndexCache sharing (research/30).

### C4. MTP draft under extended positions + windowed draft (training-free, effort low for the check, medium for the window)

Why: vLLM #37435 shows an MTP layer that does not see the extension collapses to 0 % acceptance above the native length. Ours is a
QSA layer; verify it receives the same positions/factor and the same indexer treatment as the 12 target layers, and
measure accepted length at 250K / 512K / 1M. Windowed-MTP (2607.21535: sink + 4K window for the draft only, "+28 to +44 %"
per decode step over native MTP drafts at 1M on three hybrid families, acceptance changes bounded; lossless because the target
verifies) would remove the draft's indexer scan and host-RAM gather at 1M [read]. For us the draft already attends 2,051 cells, so the gain is
the 1/13 share of the indexer scan (~1.7 ms of ~23 ms at 1M [estimate from research/32 5.2]); decide after C1.

### C5. State check for the 36 GDN layers, then (maybe) a decay rescale (training-free, effort medium-high; only if C1 shows the state is limiting)

Why: the literature predicts state problems when the state distribution is unseen (Ruiz and Gu), the vendors' MRCR cliff is unexplained,
and UPI/LongMamba show training-free state treatments exist for Mamba. No result for GDN.
- Step 1 (observation only): per GDN layer, record state Frobenius norm and the "effective remembrance" proxy (output change when the first
  X tokens are removed, Ruiz and Gu section 3.4) at 128K, 262K, 524K, 1M on the same text. A norm that grows past the trained range is the signal.
- Step 2 (only if step 1 shows drift): scale the decay gate log-decay by 1/s_gdn for s_gdn in {1, 0.75, 0.5} above 262K (UPI-like, one multiply in the DeltaNet
  kernels) and check dNLL, MRCR, retrieval. Expect a trade against short-range accuracy; UPI reports none for Mamba hybrids [abstract].
- Alternative with no model change: NoteBank-style replay memory is harness-level and shifts the problem away from the state; not engine work now.

### Not recommended now

DroPE, HyPE/HypeNet, LongRoPE2, CoPE, SpectralShift, CST, state passing (all need training; DroPE has the best hybrid evidence, but Qwen's own NoPE test of this
architecture showed endless generation after post-training); EME of Mechanics 1.1 (needs NoPE attention layers); SSMax/log-n
(our softmax size is constant); StreamingLLM/InfLLM/TokenSelect (the indexer already does selection); DCA as a separate pattern machine
(Jet-Long covers the same idea; vLLM's backend is gone); the 2026 theoretical RoPE papers (diagnostics only).

### Open questions that this survey could not answer

1. Whether Qwen's published >256K numbers (RULER 93.0 at 512K-1M, MRCR 26.4 at 1M) used YaRN x4, x2 or nothing.
2. Any accuracy number for Qwen3.5+/3.8 GDN targets at 1M from a third party (Windowed-MTP publishes speed only).
3. UPI details (PDF not retrievable): which hybrids, which tasks; the Qwen-style GDN is not in the abstract.
4. How the 2026 zero-shot maps behave with a sparse-attention model; no paper combines them (MInference + DCA in Qwen2.5-1M needed repair).

---

## 6. What is new in 2026 (by month, abstract or read as marked)

- **Jan**: HyPE/HypeNet 2601.22156 [read], MrRoPE 2601.22181 [read].
- **Feb**: CoPE 2602.05258 [read]; DySCO 2602.22175 [abstract]; MiniCPM-SALA 2602.11761 (sparse + linear hybrid, HyPE, 1M on one A6000D, 3.5x at 256K) [abstract]; theory of RoPE base bounds 2602.10959.
- **Apr**: Olmo Hybrid 2604.03444 with DroPE [read]; PoST 2604.07658; HyLo 2604.24715 (upcycling to hybrids, 32x extended context, 2M tokens in vLLM, 10B-token recipe) [abstract]; RoPE-perturbed self-distillation 2604.14339.
- **May-Jun**: Periodic RoPE 2605.27980; unified critical-scaling criterion 2605.12697; DeepSeek-V4 2606.19348 (1M trained, CSA/HCA, YaRN 16 baked in) [read]; Randomized YaRN 2606.23687; Qwen3.8 tech report with QSA RULER/MRCR (Table 3) [read].
- **Jul**: Jet-Long 2607.07740 (NeurIPS 2026) [read]; Windowed-MTP 2607.21535 [read]; HiLS 2607.02980 [read]; "Relative positions generalize, absolute positions memorize" 2607.18759.
- **Aug-Sep**: Cracks in the Foundation 2608.10296 (four small architecture choices in dense models, up to -47 % long-context extensibility, invisible in short-context loss; OlmPool of 26 7B models) [abstract]; SpectralShift 2609.14320 [read]; CST 2609.09157; content-based addressing 2609.07314; RoPE Profiler 2609.39929; DaRoPE 2609.34556; position-encoding survey 2608.10021.
- **Oct**: Mechanics of Long-Context Hybrid Models 1.1 (10-07) [read]; FreeSolo (10-12) [read]; NoteBank (10-11) [read]; SuPER-Mamba (10-12) [abstract]; Kimi K3 listing at 1.05M unverified.

Venues: ICLR 2026: DroPE, critical attention scaling, UPI, InfLLM-v2 (OpenReview version); ICML 2025: LongRoPE2; NeurIPS 2026: Jet-Long, SketchSSM (research/papers.md);
EMNLP 2025: GALI, TokenSelect; ACL 2025/2026 (Findings): AdaGroPE, LaMPE, RiPRA. OpenReview ICLR/NeurIPS/ICML lists were not browsed directly
(search and OpenAlex only); an ICML 2026 sweep is missing.

---

## Sources (read)

- Jet-Long: https://arxiv.org/abs/2607.07740 ; Windowed-MTP: https://arxiv.org/abs/2607.21535 ; NoteBank: https://arxiv.org/abs/2610.15715
- Mechanics of Long-Context Hybrid Models: https://arxiv.org/abs/2610.10114 ; HyPE: https://arxiv.org/abs/2601.22156 ; Olmo Hybrid: https://arxiv.org/abs/2604.03444
- DroPE: https://arxiv.org/abs/2512.12167 ; LongRoPE2: https://arxiv.org/abs/2502.20082 ; MrRoPE: https://arxiv.org/abs/2601.22181 ; CoPE: https://arxiv.org/abs/2602.05258
- Qwen2.5-1M: https://arxiv.org/abs/2501.15383 ; Qwen3: https://arxiv.org/abs/2505.09388 ; Qwen3.8-Flash-Next report: https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf ; cards: https://huggingface.co/Qwen/Qwen3.8-Flash-Next , https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct , https://huggingface.co/Qwen/Qwen3.5-35B-A3B
- DeepSeek-V4: https://arxiv.org/abs/2606.19348 ; DeepSeek-V3.2: https://arxiv.org/abs/2512.02556 ; Kimi Linear: https://arxiv.org/abs/2510.26692 ; configs: huggingface.co/deepseek-ai/DeepSeek-V4-Flash, openai/gpt-oss-120b, moonshotai/Kimi-K2-Instruct-0905, tiiuae/Falcon-H1-34B-Instruct, ibm-granite/granite-4.0-h-small, XiaomiMiMo/MiMo-V2-Flash, zai-org/GLM-4.6
- HiLS: https://arxiv.org/abs/2607.02980 ; FreeSolo: https://arxiv.org/abs/2610.17191 ; HSA: https://arxiv.org/abs/2510.17196 ; SpectralShift: https://arxiv.org/abs/2609.14320 ; Ruiz and Gu: https://arxiv.org/abs/2507.02782 ; Samba: https://arxiv.org/abs/2406.07522
- Issues: https://github.com/ggml-org/llama.cpp/issues/27756 , /27090 , /pull/31371 , /pull/30206 ; https://github.com/vllm-project/vllm/issues/37435 , /49296 , /62776 ; https://github.com/sgl-project/sglang/issues/36807
- Abstract only: UPI https://openreview.net/forum?id=MjmORKLHUI ; LongMamba https://arxiv.org/abs/2504.16053 ; Mamba Modulation https://arxiv.org/abs/2509.19633 ; DySCO https://arxiv.org/abs/2602.22175 ; PoST https://arxiv.org/abs/2604.07658 ; SSMax https://arxiv.org/abs/2501.19399 ; critical scaling https://arxiv.org/abs/2510.05554 ; LongPPL https://arxiv.org/abs/2410.23771 ; NoLiMa https://arxiv.org/abs/2502.05167 ; HELMET https://arxiv.org/abs/2410.02694 ; controlled study https://arxiv.org/abs/2409.12181 ; Cracks in the Foundation https://arxiv.org/abs/2608.10296
- Own: research/06, 29, 30, 31, 32 (YaRN specifics, memory, time), measurements/2026-10-28-long-context, 2026-10-29-indexcache
