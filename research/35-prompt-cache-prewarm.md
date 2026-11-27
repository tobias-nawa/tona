# 35 — Pre-warming the prompt cache: when coding agents pay the full prefill, and what we can compute ahead

Agent research (Sonnet), 2026-10-29 (issue #79)

Method: web search and page retrieval (vendor docs, arXiv abstract/HTML/PDF, GitHub discussions and issues), plus a read of
our own code (`engine/server/prompt_cache.cpp`, `prefill.cpp`, `checkpoint_file.h`) and measurements. Nothing was measured or
built here, no engine code was changed. Marking per entry:
- **[read]**: page or PDF text read for the statement.
- **[abstract]**: abstract page only.
- **[snippet]**: only a search-result snippet seen (not verified).
- **[code]**: read in our source.
- **[estimate]**: own reasoning, not measured.

Rates are named: **prefill** on our platform is ~1,300-1,700 tok/s (8K = 5 s, 64K = 43 s, 250K = 3:14 min). Numbers from
other systems are the authors' on their hardware.

---

## 0. Summary

1. **Our cache is already good for a running session** (RAM stash, 16 conversations, checkpoints at the start of the last
   message, the fork and the fixed tail). It is **purely reactive**: a checkpoint exists only as a side effect of a request
   that paid for the prefill once [code]. Nothing can be computed ahead, and nothing can be requested by a harness.
2. **Agents pay the full prefill in five situations** (section 2): first request of a session in a project the cache has not
   seen, after a pause longer than we keep the state (not our problem: we keep it for days on SSD, the cloud only for
   5 min / 1 h), after compaction (the conversation layer, but not the system prompt), after any change early in the prompt
   (tool set, cwd, git snapshot, harness version), and when side requests (checker, title, subagent) evict state.
   TraceLab (4,300 sessions of Claude Code and Codex): 95.7 % of prompt tokens come from the prefix cache, but user-initiated
   steps only 84.4 %, and the misses are caused by idle gaps > 5 min [read].
3. **Cloud pre-warming is an established, documented pattern**: Anthropic documents `max_tokens: 0` to write the cache
   without generating [read]. An identical contract on our server costs nothing in the API and every harness can use it.
4. **Recommended (section 4):** (A) a warm-up request contract (`max_tokens: 0` on both APIs, plus an explicit
   `X-IE-Warm` priority class that yields to real requests), (B) automatic startup/idle warm-up of a configured list of
   *prefix templates* (system prompt + tools per harness), stored as SSD checkpoints, (C) project-open warm-up of the stable
   head only (system prompt + tools + CLAUDE.md), triggered by the first request of a session, not by guessing. Not
   recommended now: position-independent reuse of repo files (HYPIC/LinearKV): possible in principle for DeltaNet, but
   needs a stored transition operator per segment and a quality measurement (section 3d).

---

## 1. How our cache works today [code]

Sources: `engine/server/prompt_cache.cpp` (137 lines), `prefill.cpp`, `checkpoint_file.h`, README "Assumptions", docs/use/environment.md,
measurements/2026-10-24-conversations, 2026-10-26-prefill-slices.

- **Hit logic.** `pick_source` takes the longest exactly matching beginning of the new prompt among: the running tokens
  (VRAM), the checkpoint of the previous prompt end, the RAM stash (`stash_`), and then the SSD files (`load_disk_checkpoint`:
  longer match than everything in RAM wins, file is read into the stash). The match is exact token-prefix (`prefix_of`).
  A hybrid state cannot be rewound, so only stored positions are usable.
- **Where checkpoints are set.** (1) End of the prompt (stash of the running conversation, RAM only). (2) Start of the last
  message, if at least 4,096 tokens had to be computed before it. (3) `fork`: the longest common prefix with a stashed or
  slot-resident prompt, if it is at least 2,048 tokens beyond the best hit (minus a 16-token seam margin). (4) `tail`: the
  fixed end behind growing content (auto-mode checker), only RAM, it moves with every request. `plan_checkpoints` creates
  (2)-(4) while prefilling; a checkpoint costs 60-180 ms to stash.
- **Storage.** RAM: up to `IE_PROMPT_SLOTS` (16) conversations and `IE_PROMPT_RAM_GB` (12). SSD: only (2) and (3), not the
  moving tail and not the plain prompt end (`checkpoint_at` writes with `disk_write` for `!tail`); `IE_PROMPT_CACHE` directory,
  `IE_PROMPT_CACHE_GB` (16). File name = fingerprint + hash of tokens; fingerprint covers all tensors, state sizes, KV row
  and `DISK_VERSION`; size and tokens are checked before loading, a broken file is deleted.
- **Eviction.** RAM: entries fetched before and used in the last 16 operations are protected, then oldest first (Marconi idea;
  an end that continues an older one inherits its hits). SSD: oldest by modification time (renewed on read) until the size
  limit holds. No decay by age, so the SSD cache survives restarts and days.
- **Size.** 134 MB DeltaNet state + 84 MB drafter ring + ~13.9 KB per token (KV 12.9 + pool 1.5) (measurements/2026-10-24-conversations):
  16K = 459 MB, 100K = 1,763 MB. Store/fetch 100-330 ms.
- **Isolation.** Only by token equality: any client that sends the same token prefix gets the same state. There is no
  per-user or per-project key (the `X-IE-Project` header only affects slot choice).

**Gaps for pre-warming:**
1. No way to ask for a checkpoint without generating (every checkpoint rides on a real request). With `max_tokens` the first
   token is always sampled; whether `max_tokens: 0` is accepted was not checked in code (OpenAI default is unbounded,
   Anthropic default 4096; `verify.cpp` stops at `++n >= max_tokens` after the first token).
2. The SSD only receives (2) and (3). A first request in a new project writes the start-of-last-message checkpoint, which
   includes cwd, git snapshot and CLAUDE.md, so it does not help the next *project*; the shared head (system prompt + tools)
   only gets a checkpoint when a second, diverging conversation is seen (fork). Before that, the first session of every
   project pays the whole head again.
3. Nothing is ever computed while the GPU is idle.
4. The prefill is in slices of 4,096 tokens next to a running decode, but an idle-time job has no priority class below a
   real request.

---

## 2. When coding agents pay the full prefill

### 2.1 Measured/read evidence

| Finding | Source |
|---|---|
| ~4,300 sessions, 350K LLM steps (Claude Code + Codex): 95.7 % of prompt tokens from prefix cache (Claude 95.8, Codex 95.7). User-initiated steps 84.4 % (Claude 86.9, Codex 78.2), tool-result steps 97.5 %. Median step: ~119K prefix tokens, 875 append, 214 output | TraceLab, arXiv 2606.30560, Tables 1/11 [read, PDF text] |
| Misses concentrate after idle gaps: "when the gap is larger than 5 minutes, low-hit-rate steps begin to appear, and after 1 hour almost all steps miss". Misses cause 3.8x more prefilled tokens than truly fresh tokens; fresh tokens are only 19.0 % of append tokens (12.3 % Claude); user-initiated steps carry 1.7 % fresh tokens (prefill amplification 58.8x Claude, 22.3x Codex). Removing thinking-time misses would cut append tokens by 45.9 % | TraceLab 7.2/7.3, Table 12/13 [read] |
| Eviction-time sweep: 1 min -> 1 h lifts achievable hit rate from 85.4 % to 98.6 %; at 5 min already ~94 % | TraceLab 7.4 [read] |
| Claude Code cache layers: system prompt (instructions + tool definitions) -> project context (CLAUDE.md, auto memory, unscoped rules, reloaded at session start, `/clear`, `/compact`) -> conversation. A change in a layer invalidates it and everything after it | Claude Code docs "How Claude Code uses prompt caching" [read] |
| Invalidating actions: model switch, effort change (older models), fast mode, MCP server connect/remove when tools are loaded upfront, plugin MCP, deny rule on a tool without tool search, `/compact`, image trimming, Claude Code upgrade | same [read] |
| Compaction: summary replaces history, "no shared prefix with the old one"; system prompt + project context can stay cached; the summarization request itself reuses the old prefix | same [read] |
| Cache scope: the conversation opens with an announcement of cwd, platform, shell, OS version; the system prompt embeds auto-memory paths; "two sessions in different directories build different prefixes". Sequential sessions share the prefix only when the git status snapshot taken at startup matches (branch + recent commits are in the conversation) | same, "Cache scope" [read] |
| Subagents: own system prompt and tool set, "first request doesn't read the parent's cache"; a fork inherits everything exactly and reads the parent's cache | same, "Subagents and the cache" [read] |
| Attribution block at the start of the system prompt carries client version and a prompt fingerprint; before v2.1.181 it included a per-request token on custom base URLs. `CLAUDE_CODE_ATTRIBUTION_HEADER=0` omits it | Claude Code env-vars docs [read] |
| llama.cpp users with Claude Code: "forcing full prompt re-processing" every few requests with prompts of tens of thousands of tokens, fixed (qualitatively) by `CLAUDE_CODE_ATTRIBUTION_HEADER=0`; no before/after timings | blog, mykolaaleksandrov.dev 2026-06 [read, weak] |
| llama.cpp hybrid models: #18497 (Qwen3-Next full reprocess, closed per a roundup), #24587 (strict checkpoint validation deletes checkpoints and forces full reprocess), #22450 (Qwen3.6 hangs in multi-turn agent flows), #19794, #19977 listed open in a roundup; "architectural, complete fix not merged" | GitHub issue titles via search; status from the Zenn roundup [snippet, status not checked] |
| Cloud keepalive: idle gap > TTL kills the cache. Aider `--cache-keepalive-pings N` pings every 5 min (off by default); one blog measured 0 of 48 warm at 10 min without keepalive, 40 of 40 with | aider.chat docs [snippet], blog.mempko.com [snippet] |
| Own measurements: auto-mode checker (~30K-token fixed head + growing log) forced 45-90 s per switch before conversations; now 1.7 s. Claude Code startup prompts: ~20K interactive, Codex ~11K, Crush ~12K, Qwen Code 14-15K | research/10, 07; measurements/2026-10-24-conversations |

### 2.2 What this means for us, by situation (prefill at ~1,500 tok/s)

| Situation | Who pays | Our state today | Cost |
|---|---|---|---|
| **First request of a session in a new project** | Everyone. System prompt + tools (~11-20K) + CLAUDE.md + cwd/git block. The cwd line sits *before* most of the conversation and is unique per project | Fork checkpoint only after a second diverging conversation; otherwise full | 20K = 13 s; 30K = 20 s |
| **After server restart / long pause** | Cloud: after 5 min / 1 h. Us: no TTL, SSD survives restart | SSD hit for (2)/(3) checkpoints; the plain prompt end is not on SSD, so the *last turns* since the checkpoint are recomputed (usually small) | usually a few K tokens |
| **After compaction (`/compact`, auto-compact)** | Conversation layer. The summary is new text after system prompt + CLAUDE.md. The summarization request itself is a long, normal continuation | The head (system+tools+CLAUDE.md) is only reusable if a checkpoint sits there: (2) is at the *last message* start, i.e. after the summary, not at the project head | the summary (5-20K) plus head if no fork checkpoint: 4-26 s |
| **Early change** (tool set, MCP server added, harness upgrade, model alias change, attribution fingerprint, git snapshot differs, date in system prompt) | Everything after the change | No checkpoint before the change unless a fork was seen | full prefill of the conversation, 64K = 43 s |
| **Side requests** (checker, titles, memory, summaries) | Own prompt, evicts single-slot state | Solved by 16 conversations + tail/start-of-message checkpoints (1.7 s) | solved |
| **Subagents** | Own system prompt and tools, first request cannot hit the parent | The subagent *head* (system prompt + tools) is identical across all subagents of one type; hit only if a checkpoint exists there | 5-10K = 3-7 s per new subagent type, once |
| **Parallel sessions** | Same project/dir builds matching prefixes (Claude Code docs); different dirs do not | Slots choose the free slot with the longest match; fork checkpoint helps (3.1 instead of 6.9 s measured) | covered |

**Conclusion.** The unavoidable fresh tokens (new user input, tool results) are 19 % of append tokens; the rest can in
principle be served from cache [TraceLab, read]. For us the remaining *avoidable* cost is mostly the **head of a session**
(system prompt + tools + project context), which is identical across sessions up to the cwd/git block, and the head of
**subagents**. Both are stable per harness version and can be computed before the user sends anything.

---

## 3. Pre-warming strategies: state of the art

### 3a. System prompt and tool definitions ahead of time (per harness)

- **Anthropic API** documents it: `max_tokens: 0` writes the cache at every `cache_control` breakpoint and returns an empty
  `content` array with `stop_reason: "max_tokens"` and a full `usage` block; no output tokens billed; rejected with
  `stream: true`, extended thinking, structured outputs, forced `tool_choice`, and in batches. TTL applies, so a pre-warm every
  5 min or the 1 h TTL [Anthropic prompt caching docs, read]. Place the breakpoint on the last block shared with the follow-up.
- **Community pattern**: pre-warm when a request is likely (user opens app), not for single-shot scripts; hit probability times
  latency saved must exceed miss probability times speculation cost (blog, tianpan.co 2026-05 [read], weak source).
- **Aider** keepalive pings (`--cache-keepalive-pings`) keep a cloud cache alive [snippet].
- **llama.cpp**: `--slot-save-path` + `POST /slots/{id}?action=save|restore` with a file name; manual, client decides
  [GitHub discussion #18244, read]. Caveats from the thread: model, quant, flash-attention setting and KV type must match;
  "restore can report success but still silently re-prefill" (check `prompt_n`); no format guarantee between builds.
  Auto-persist feature request closed as not planned [snippet]. Host-RAM prompt cache `--cache-ram` (PR #16391) acts as extra slots by
  prefix similarity [snippet]; `--cache-idle-slots` saves a slot's state when it switches tasks [snippet, semantics unverified].
  No `n_predict: 0` warm-up documented in anything we found.
- **Ollama**: an empty request to `/api/generate` or `/api/chat` preloads the *model* and `keep_alive` controls residency
  (default 5 min) [FAQ via search, snippet]. That warms weights, not a prompt prefix.
- **vLLM / SGLang / LMCache**: automatic prefix caching; SGLang HiCache extends RadixAttention to host memory and storage
  with a prefetch threshold (default 256 tokens) [snippet]; LMCache has a local-disk backend. None of these has a "warm this
  prefix" feature specific to coding harnesses; the pattern is the same as ours (store what was computed). LMCache does not
  support Gated-DeltaNet hybrids in its multiprocess connector (research/10).

**Assessment.** The pattern (an explicit zero-output request) is standard and cheap. The new part for us is that our
checkpoint is a *file on SSD that survives restarts*, so the warm-up can be a one-time setup per harness version, not a
recurring 5-minute keepalive.

### 3b. Project files and repo context on opening a project

- **Indexing (Cursor/Continue) is retrieval, not KV precompute**: embeddings of code chunks, nothing stored as model state.
  We found no source that pre-computes KV for repository files [no source found; not verified for Cursor/Continue internals].
- **Speculative Prefill** (arXiv 2502.02789, ICML 2025) [abstract]: a small model scores token importance so the main model
  prefills only a subset; 7.66x less TTFT on Llama-3.1-405B. It reduces the prompt, it does not precompute before the request,
  and it changes the prompt (quality risk for code, where every identifier matters). Not a fit for exact agents.
- **"Prefill while typing"**: no arXiv paper found. A defensive publication (Technical Disclosure Commons, 2026-08) describes
  firing speculative requests during typing and reusing shadow-fetched context [snippet]; a blog describes "speculative
  context loading" [read]. Both are about retrieval context, not KV.
- **Why it is weak for coding agents**: the user message is 1.7 % fresh tokens of the append (TraceLab) and the agent's own
  next step depends on the model's output; what repeats is the *history*, which is already prefix-cached once computed. The
  unpredictable part (which files the agent reads) cannot be guessed reliably.
- **What is predictable**: the head of the prompt (3a) and, on resume, the previous conversation (already on RAM/SSD for us).

### 3c. API extensions

| Mechanism | Pros | Cons |
|---|---|---|
| `max_tokens: 0` (Anthropic-documented) on `/v1/messages` and `/v1/chat/completions` | Zero harness changes where the harness (or a wrapper script) can send a request; same semantic as the cloud | Harnesses do not send it by themselves |
| llama.cpp-style slot save/restore endpoints | Explicit, files portable | Manual, build-sensitive format, client needs to know file names; we have fingerprint validation already |
| Dedicated endpoint (e.g. `POST /v1/ie/warm` with the same JSON as a chat request, returns usage + checkpoint position) | Can carry priority, TTL/pinning, project key; no generation path | Not standard; only our own tooling/`ie-top`/wrapper uses it |
| Server-side template list (startup/idle warm-up) | No harness cooperation at all | Needs the exact bytes the harness will send: tools JSON, system text, rendering by our chat template |

### 3d. Reusing KV for non-prefix parts (position-independent caching, PIC)

For full-attention models:
- **CacheBlend** (selective recompute of high-deviation tokens, dynamic), **EPIC** (arXiv 2410.15332: static sparsity, recompute
  the first tokens of each chunk; up to 3x TTFT vs CacheBlend, accuracy loss within 7 %; CacheBlend's overhead 16.3-63.6 % of TTFT)
  [abstract/snippet]. **KVLink** (trained linking tokens), **Block-Attention** (blocks trained to attend independently),
  **MEPIC** (page-aligned), **MiniPIC** (NeurIPS 2026, <100 LOC primitives), **SemPIC** (learned adaptation, 2607.28069) [snippets].
  Several need fine-tuning (KVLink, Block-Attention).

For DeltaNet/Mamba hybrids (our model):
- The recurrent state at position *n* depends on all tokens before it, so a chunk computed alone yields a state for the wrong
  history. Two 2026 papers address exactly this:
  - **HYPIC** (arXiv 2607.01299, SGLang, 14K lines): store with each segment a **segment-cumulative transition operator** next to the
    zero-start end-state, then compose independently cached segments **near-exactly** in constant time per segment length. For
    dense-transition models (DeltaNet, GDN, KDA) the operator is a d_k x d_k matrix per head per layer (32 KB at 128 x 128 fp16, same
    element count as the state); building it adds 5.2-6.7 % to the segment prefill (Qwen3.5-35B-A3B); applying costs
    O(d_k^2 d_v). Full-attention layers are repaired by recomputing a **seam window of w = 8 tokens** per interior segment.
    3.25x average TTFT reduction vs prefix cache, 1.66x QPS, quality 1.71 points behind full recompute on average (Ring-mini 3.44,
    Ring-flash 3.29, Qwen3.5-35B +0.47, Qwen3.5-122B -0.56); "naive addition" loses 66.9 % of full-recompute score [HTML read for
    the operator parts; headline numbers from abstract].
  - **LinearKV** (arXiv 2608.11231): one cached state as the initial state of the linear layers is enough; on two GDN models
    equal to exact composition (up to 92 % of full quality), on Mamba-2 exact composition collapses (46.6 % vs 86.8 %); TTFT
    0.46x of full prefill; LongBench QA and RULER 8K-32K [abstract].
- **Fit for us:** only attractive for text that appears at *different positions* in different prompts: a repeated file read in
  two sessions, retrieved RAG chunks. Coding harnesses read files by tool calls inside the conversation, in different order and
  surrounded by different text. The gain is unmeasured for that case, the quality evidence is on QA and summarization at
  8K-44K, our model additionally has the sparse QSA indexer pool and int8 KV with RoPE keys (re-rotation as in research/32), and the
  DeltaNet state of 36 layers is 134 MB so one operator per segment would roughly double the checkpoint file per segment (a
  d_k x d_k matrix has the same size as the d_k x d_v state when both are 128 [read]). **Not recommended before A and B are in place;
  needs its own research ticket with a quality measurement (rule 3).**
- **Tail-Replay** (arXiv 2608.30310, in research/10): store only attention KV, rebuild the DeltaNet state from a short suffix
  replay (approximate). Reduces SSD volume (no state), not needed for warm-up.

---

## 4. Proposal for us

### 4.1 Mechanisms (ranked)

**A. Warm-up request contract: `max_tokens: 0` plus an explicit class.**
- Accept `max_tokens: 0` on both APIs as "prefill, checkpoint, return usage with no content", identical to the Anthropic
  contract (empty content, `stop_reason: max_tokens`, usage block). Reject with `stream: true` and forced tool choice like
  Anthropic does, or simply allow streaming and send the final usage; our first-byte rules (keepalive during prefill) apply anyway.
- A request header `X-IE-Priority: warm` (or a field in the extension namespace) puts it below auto-mode checker, main and
  subagent in the queue and lets it **yield**: at every 4,096-token slice boundary it checks the queue (the slicing exists, README)
  and is abandoned or paused when a real request arrives. The state computed so far can be kept as a checkpoint.
- Writes a checkpoint at the end of the prompt *and* directly to the SSD (the plain prompt end is not on SSD today).
  A new `ckpt` flavor "head" that is never evicted by the moving-tail logic and is pinned against `disk_prune` below a quota.
- How a harness triggers it: nothing built-in does `max_tokens: 0` against a local server today [no source found]. Realistic paths:
  (i) a one-line wrapper/hook in the harness startup (Claude Code SessionStart hook; Codex/OpenCode via script), (ii) our own
  `ie` command `ie warm --harness claude-code --cwd .` that builds the request the harness would send, (iii) automatic (B).
- Expected effect: first-token for the first request of a session goes from 13-20 s (20-30K head) to the checkpoint load time
  of ~0.1-0.3 s plus the tokens after the head; **estimate**, uses our measured fetch times for stashed conversations, not
  measured for a warmed head.

**B. Template warm-up at server start and when idle.**
- Config: `IE_WARM_DIR` with request bodies (or harness profiles: `claude-code`, `codex`, `opencode`) captured with the existing
  `IE_DUMP_DIR`; the server replays them with the warm class when idle (after startup, after update, after a fingerprint change
  invalidated files). Re-runs only when the file for these exact tokens is missing. The model fingerprint check already throws out
  stale files; `DISK_VERSION` bumps trigger a re-warm in the background.
- Which prompts: the system prompt + tool block of each harness and version, and the head of each subagent type (Explore,
  general, Plan for Claude Code; same tools and text for all subagents of the type), plus the auto-mode checker head
  (~30K tokens, identical across sessions; we measured it). Cut the stored prefix **before** the first variable item
  (cwd, git snapshot, date, memory paths). Claude Code docs state cwd/OS/shell are announced at the start of the conversation and
  the git snapshot is in the conversation; the exact position must be taken from a dump (not verified) [docs read].
- Harness-side levers that make the head stable (harness settings, no code on our side): `CLAUDE_CODE_ATTRIBUTION_HEADER=0`
  (otherwise the fingerprint at the start of the system prompt may differ per version/request depending on the version [docs read]),
  `CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS=1` removes the git snapshot [docs read]. Ship these as part of the client profiles
  (research/07 item 7).
- **Detect the cut point automatically (preferred over hard-coded):** with two requests of the same harness the longest common
  prefix is the stable head; the existing `fork` checkpoint does this after the second session. Extend: when the fork exists in
  RAM, write it to SSD and mark as `head`; warm-up then only needs to *replay* the head once after a model/version change.
  This reduces B to "keep what the fork logic already found, and protect it".
- Expected effect: removes the head cost for the first session in every new project and for every new subagent type; removes
  the 4-26 s head after compaction when the head checkpoint exists. **Estimate** from section 2.2 prefill times.

**C. Project-open warm-up of the stable project head, triggered by the harness's first request (not by guessing).**
- When the first request of a session arrives, the server already computes it; the new part is that the **checkpoint at the end of
  system+tools+CLAUDE.md** is written to SSD immediately (not only at the start of the last message). The next session in the same
  project (same CLAUDE.md, same cwd) and the post-compaction request then start from this point. Concretely: plan_checkpoints gets
  a third cut point "end of the first user-visible project block" found by comparing with the previous request.
- This is cheaper than (b) in 3b: no speculation about files; uses the 19 % / 81 % split of TraceLab: predicted content is only
  what was already seen.
- Optional later: pre-read the *recently touched* files at project open as a normal prompt, i.e. only if the same project has a
  history; speculative; not recommended without a measured hit rate.

**Not now:** HYPIC/LinearKV-style PIC (3d), Speculative Prefill (changes the prompt), embedding-style indexing (not a prefill
problem).

### 4.2 Memory per warmed prefix

Per prefix (measurements/2026-10-24-conversations): 134 MB DeltaNet state + 84 MB drafter ring + ~13.9 KB per token.

| Prefix | State+ring | KV+pool | File/RAM | Prefill saved at 1,500 tok/s |
|---|---:|---:|---:|---:|
| 11K (Codex head) | 218 MB | 153 MB | 371 MB | 7 s |
| 20K (Claude Code interactive) | 218 MB | 278 MB | 496 MB | 13 s |
| 30K (auto-mode checker head) | 218 MB | 417 MB | 635 MB | 20 s |
| 64K | 218 MB | 890 MB | 1.1 GB | 43 s |

[estimate: arithmetic from the measured per-token and fixed sizes.] The drafter ring (84 MB) is needed for MTP drafting after
the checkpoint; whether a head-only file could drop it and recompute it from the last tokens is a code question, not answered here.
A realistic warm set (3 harnesses x ~3 versions x 4 subagent types, mostly sharing the head per harness) is 5-15 files =
2-8 GB on SSD, inside the existing 16 GB budget if `head` files are quota-protected; RAM stash is untouched because warm files
are loaded from SSD on demand (100-330 ms).

### 4.3 Risks

1. **GPU busy when the user asks.** The warm job must be preemptible at slice boundaries (4,096 tokens = ~2.7 s at 1,500 tok/s;
   too coarse for an interactive request? The user waits up to one slice; smaller slices cost prefill speed, 2,048 gave ~25 % slower
   prefill, measurements/2026-10-26-prefill-slices). Priority: warm < subagent < main < checker, and warm only starts when no
   request ran for N seconds. **Never** use a blocking HIP call while the megakernel waits (AGENTS.md GPU safety).
2. **Desktop load.** The desktop shares the GPU (1.2-1.9 GB VRAM). A background prefill of 20-30K tokens occupies the GPU for 13-20 s;
   run warm-up only at startup/idle with a duty-cycle limit, and not during `IE_PREFILL_YIELD_MS`-style yielding situations. Needs a
   measurement of desktop frame stalls before enabling by default.
3. **Security and isolation.** The cache is keyed by tokens only: any local client sending the same prefix gets the stored state
   (timing side channel: a hit reveals that the prefix exists, like in cloud caches). For a single-user local server that is
   acceptable; for several users or projects with secrets in CLAUDE.md the SSD files contain a function of private text (KV is
   invertible in principle). Mitigations: file mode 0600 in a per-user dir; `head` checkpoints only for **templates** (no
   project text); per-project key in the file name only if multi-user is a goal; warm endpoint bound to loopback or same auth as
   the API. Do not warm from other users' dumps.
4. **Staleness.** Harness upgrade changes the head (Claude Code docs: upgrade rebuilds the prompt) so warm files become useless but
   still occupy quota: eviction must prefer unused `head` files; re-warm after dump comparison.
5. **Exactness.** A warm-up must go through the *same* tokenizer/chat rendering as real requests; a one-token difference at
   the seam kills the hit (existing 16-token margin handles merge differences at the cut). Test: warm, then real request, assert
   `prompt_n` of only the new tokens.
6. **Determinism.** Checkpoints are bit-identical to the same prefill path only if the pass order matches (README rule 3; slices of
   4,096 vs one piece may differ in float order; measurements/2026-10-23 showed unequal answers for a repeated prompt computed fresh).
   A warmed head must be checked against a cold prefill with `IE_DETERMINISTIC=1`.
7. **Client timeouts.** `max_tokens: 0` without streaming blocks for the whole prefill; harness wrappers need a long timeout or a
   fire-and-forget form.

### 4.4 Test plan (not run)

1. Dump the first request of Claude Code, Codex, OpenCode (`IE_DUMP_DIR`), find the stable head by diffing two sessions in different
   projects; record the token index of the first difference and the head size per harness.
2. Implement A only (accept `max_tokens: 0`, write the end-of-prompt checkpoint to SSD) and measure first-token time of the next
   session cold vs warmed, 3 runs each, spread reported (rule 2).
3. Measure desktop impact of a 20K-token warm-up on the GPU (frame times, `IE_PREFILL_YIELD_MS`) before B is idle-triggered.
4. Check the bit-identity of a warmed head vs cold prefill with `IE_DETERMINISTIC=1`.

---

## 5. Open questions

- Does `max_tokens: 0` already work in the server? (not checked in code; the verify loop stops after the first sampled token).
- Where exactly is the first variable item in Claude Code/Codex/OpenCode requests (cwd line, date, git snapshot)? Needs the dump.
- What fraction of our real sessions begin with a head the cache has already seen? We have no log statistic for it; the
  fork logic counts only when a second conversation diverges.
- How strong is DeltaNet-hybrid PIC for repeated file reads in coding sessions? Nobody has measured it (HYPIC/LinearKV evaluate QA
  and summarization).
- TraceLab section on KV retention (85.4 % to 98.6 % hit rate for 1 min to 1 h) was measured for cloud-scale batches; for a single
  local user the "retention time" is limited only by SSD quota, so the claim translates directly (SSD checkpoints do not expire).

## Sources

- Anthropic prompt caching docs (pre-warming, TTL, hierarchy): https://platform.claude.com/docs/en/build-with-claude/prompt-caching [read]
- How Claude Code uses prompt caching: https://code.claude.com/docs/en/prompt-caching [read]
- Claude Code env vars (attribution header, git instructions): https://code.claude.com/docs/en/env-vars [read, first 100K characters]
- TraceLab, arXiv 2606.30560: https://arxiv.org/abs/2606.30560 [PDF text read]
- HYPIC, arXiv 2607.01299: https://arxiv.org/abs/2607.01299 [HTML read]
- LinearKV, arXiv 2608.11231: https://arxiv.org/abs/2608.11231 [abstract]
- EPIC, arXiv 2410.15332: https://arxiv.org/abs/2410.15332 [snippet]
- Speculative Prefill, arXiv 2502.02789: https://arxiv.org/abs/2502.02789 [abstract/snippet]
- llama.cpp discussion #18244 (slot save/restore): https://github.com/ggml-org/llama.cpp/discussions/18244 [read]
- llama.cpp PR #16391 (host-memory prompt cache), issues #18497, #24587, #22450 [snippets, status not checked]
- Claude Code with llama.cpp cache fix: https://mykolaaleksandrov.dev/posts/2026/06/claude-code-llamacpp-prompt-cache-fix/ [read, weak]
- SGLang HiCache: https://lmsys.org/blog/2025-09-10-sglang-hicache [snippet]; LMCache arXiv 2510.09665 [snippet]
- Aider caching docs https://aider.chat/docs/usage/caching.html [snippet]; keepalive measurements https://blog.mempko.com [snippet]
- Context priming/pre-warm notes: https://tianpan.co/blog/2026/05/07/speculative-context-loading-ai-feature-latency [read, weak]
