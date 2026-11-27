# Research 70: do Ornith-1.5-35B-A3B or Bolt-24GB handle Rust/Go better? (2026-11-19)

## Question
Is Ornith-1.5-35B-A3B (Q4_K_M) or Nemotron 3.5 Bolt 74B-A9B (24GB variant) a better Rust/Go target than the
pruned ISTA Coder IQ1_M (Rust 0/10, Go 1-4/17)? Web-only; HF discussion bodies read via WebFetch (small model summaries).

## Ornith-1.5-35B-A3B
Measured by vendor (model card, five-run averages, Terminus-2 / own harness, no dates on page):
https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B
- SWE-bench Verified 79, SWE-bench Pro 59.6, SWE-bench Multilingual 71.4 (aggregate, no per-language split),
  Terminal-Bench 2.1 67.8, NL2Repo 46.2. Qwen3.6-35B-A3B in the same table: 73.4 / 49.5 / 67.2 / 52.5 / 29.4.
- Not reported on the card: Aider polyglot, MultiPL-E, LiveCodeBench, any Rust or Go number. Not found anywhere else
  (independent): no Aider polyglot, no Go number.
- Aggregators repeat the vendor numbers (benchlm.ai), not independent.
Independent-ish (third party, derivative model): ManniX-ITA CoderX card, MultiPL-E, 300 problems, Q6_K+imatrix:
https://huggingface.co/ManniX-ITA/Ornith-1.5-27B-A3B-CoderX
- Base Ornith 35B (256 experts): Rust 0.75, Java 0.91, JS 0.90, overall 0.853. Their 184-expert prune: Rust 0.71-0.74.
  No Go in that set. Rust is the weakest of the three languages for every variant, but 0.75 is far above our 0/10.
  Caveat: MultiPL-E is single-function HumanEval-style, much easier than Exercism with 4 attempts.
Language dropped? Card names no training data mix (only "continued pretraining, mid-training, RL" on Qwen3.5/Gemma4).
Nothing suggests a dropped language. Note the ManniX prune uses a "per-category competence map" (not ours to use).
User reports (all HF discussions on ornith-ai/Ornith-1.5-35B-A3B-GGUF, https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/discussions ,
26 open, dates only relative on page, roughly Aug-Oct 2026):
- #6 "Not for coding?": one user, Q8, single-file HTML 3D aquarium, runtime errors, v1.5 worse than 1.0 and Qwen 27B.
  Replies: try Q6, check sampling; no resolution. Open. One anecdote, not Rust/Go.
- #5 excessive CoT verbosity on simple tasks (open); #19/#20 positive. No Rust or Go report found anywhere.
Q4_K_M quality: no coding-quality measurement found. Only a KL-divergence figure (Q4_K_M 0.0728 vs Q8_0 0.0136) from
a third-party "uncensored" derivative (https://atomic.chat/blog/guides/how-to-run-ornith-1-5-uncensored-locally),
not the vendor model. Imatrix questions open in discussions #8/#10. Quality thus unknown for Q4_K_M; Q5/Q6 exist.

## Bolt (incl. the 24GB variant)
Identity (NVIDIA first-party, not a community quant): "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB" is the release itself.
https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Bolt-24GB/ and
https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF
- Compressed from Nemotron 3.5 Super VL (120B-A12B) with "Star Elastic" router-guided architecture search: experts per
  layer and widths pruned, top-K 22 -> 9, ~120B tokens recovery distillation (8K then 128K), vision restored, RL for
  agentic/MCP/SWE, then quantization-aware distillation: experts IQ2_XXS (2.0625 bpw), Mamba linears NVFP4,
  attention/embeddings/lm_head Q8_0.
- So it IS expert-pruned (and 2-bit). Calibration data: not specified on either page. No statement about languages.
  The technical report PDF (research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-Technical-Report.pdf)
  could not be text-extracted by my tool (compressed streams); it may name the data. Not read: gap.
- LiLiCorr file (1.0 GB): NVIDIA's parallel drafter for speculative decoding, 1.13B params, mean acceptance length 4.68
  greedy at 7 draft tokens (vLLM, concurrency 64). Not a quality patch; irrelevant to Rust/Go quality.
- llama.cpp: model card says GGUF is meant for llama.cpp, "latest, build from source", no minimum version given;
  recipes for 32 GB and 28 GB. Discussion #2 asks about a missing SPEC_DRAFT. Whether OUR checkout loads it: not tested.
Benchmarks (vendor, HF card, BF16 competitors, Bolt IQ2): Terminal-Bench 2.1 50.98 (Qwen3.6-35B 44.38), GPQA 86.43,
BrowseComp 66.75, MCP-Atlas 52.20. No SWE-bench, Aider, MultiPL-E, LiveCodeBench, Rust or Go number for the 24GB model:
not found. Parent Super VL: SWE-bench Verified 71.1 self-reported via aggregator (benchlm / benchmarklist), unconfirmed;
no per-language data. Full 74B-A9B-before-quantization numbers: not found.
User reports (HF discussions on the GGUF repo, https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/discussions ,
9 open, 0 closed, no maintainer replies seen):
- #6 (about 27 days before 2026-11-19, bigei60): Go FIFO queue, Aider 0.86.2 + llama.cpp, RTX 3090. First pass did not
  compile (missing `sync` import); after one repair it compiled but 2 tests failed (Pop bug). Earlier Go TTL-cache
  attempt also had compile errors. Qwen3-Coder passed the FIFO task first try (not controlled). Author says not a claim
  the weights are defective. This is the only Go-specific report; n=2 tasks.
- #5 "inconsistent and wrong ... coding or creative" (JS MCP server, broken code); #8 "Do not change if you have
  Qwen3.8 27b"; #3 looping and poor Russian. #1 positive. No Rust report found.
Other independent coding data: none found. NVIDIA's own coding table is throughput only (SPEED-Bench-Coding TPS 84 vs
Qwen3.6 97), not accuracy.

## What it means
- Neither model has Rust/Go per-language evidence from an independent source. Do not put either in the installer's
  Rust/Go recommendation on documentation alone.
- Ornith is the better bet: vendor SWE-bench Multilingual 71.4 (above Qwen3.6 67.2), third-party MultiPL-E Rust 0.75,
  no sign of pruning or a dropped language in the base model, Q4_K_M loads in stock llama.cpp (08-models.md). Risks:
  single bad coding anecdote (#6), Q4_K_M quality unmeasured, vendor is new, numbers vendor-only.
- Bolt-24GB is heavier risk: 2-bit experts, pruned (top-K 22 -> 9, experts removed), calibration undisclosed, one failing
  Go report, no coding accuracy published. NVIDIA's own pitch is agentic/browse, not coding. Treat as unproven for Rust/Go.
- Our test should measure (same harness as research/69): Aider-polyglot-style Rust and Go Exercism sets, 4 attempts,
  all 3 models plus Ornith at Q4_K_M and Q6_K to isolate quantization; greedy/recommended sampling stated; reasoning
  budget stated (Bolt #6 used 1024); Rust and Go each at least the 17 exercises we already use, several rounds for spread;
  then decide: Ornith Rust/Go pass rate clearly above ISTA (0/10, 1-4/17) -> installer points Rust/Go users to it.
  First check Bolt loads in our llama.cpp build before spending GPU time.
- Not checked: the Bolt tech report text, per-discussion replies beyond those fetched, Reddit (search found no threads).
