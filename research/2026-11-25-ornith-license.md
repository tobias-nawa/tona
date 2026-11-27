# Ornith 1.5 35B-A3B: license evidence and offer decision (checked 2026-11-25)

All HF facts via the public API / raw files with a generic user agent. No contact with anyone.

## Short answer
Yes, Tona may offer it, with an honest label. The "no licence at all" blocker does not apply: the publisher declares MIT in
four independent places, one of them a real MIT LICENSE file in its official GitHub repo (found today; the 2026-11-19 check
looked only at the HF repos). What stays unverified: the HF repos ship no LICENSE file, and the upstream (Qwen/Gemma)
attribution is not carried by Ornith.

## 1. Where the publisher states the license
| Place | Finding |
|---|---|
| HF card YAML of ornith-ai/Ornith-1.5-35B-A3B and -GGUF | `license: mit`, `license_link` to .../Ornith-1.5-35B-A3B/blob/main/LICENSE. File does not exist (raw LICENSE = 404; no LICENSE/NOTICE in `siblings`; same for 1.5-9B, 1.5-122B, 1.5-397B, 1.0-35B and the FP8/GGUF repos). Rechecked 2026-11-25, unchanged. |
| GGUF metadata, local file Ornith-1.5-35B-Q4_K_M.gguf (read-only header read) | `general.license = mit`, `general.license.link = https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE`. The file itself carries the declaration, and it is the same dead link. No other license key, no attribution text. |
| GitHub https://github.com/ornith-ai/Ornith-1 (org ornith-ai, "Ornith-1", last push 2026-10-20) | Real LICENSE file: MIT License, "Copyright (c) 2026 deepreinforce.ai" (added 2026-06-21, API license MIT). README: "Both generations [1.0 and 1.5] are **MIT licensed, globally accessible, and free from regional limitations**." This is the best evidence: full text, copyright holder named, covers 1.5 explicitly in prose. Caveat: the repo is code and docs; the LICENSE sits at repo level and the README applies the words to the model generations. |
| Ornith-1.0 HF card (ornith-ai/Ornith-1.0-35B, formerly deepreinforce-ai/...) | "Licence: MIT licensed, globally accessible, and free from regional limitations." (explicit prose). Its license_link also points to a LICENSE missing in the HF repo (deepreinforce-ai path redirects to ornith-ai). |
| Ollama library page https://ollama.com/library/ornith (copy of the card) | repeats "Licence: MIT licensed ..." (publisher text, not Ollama's own judgement). |
| Website https://ornith.ai/ornith_1_5.html | no license statement (only the base-model sentence below). Blog/paper: none beyond the blog. |
| config.json | `model_type qwen3_5_moe`, architecture Qwen3_5MoeForConditionalGeneration; no license field. |

HF discussions: Ornith-1.5-35B-A3B #36 (2026-10-14, open): a user notes the MLX-4bit repo has no `license` field and asks
whether MIT is intended; it concerns the MLX repo, not the missing file. Note: ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit
has no license tag today. No discussion in the 35B-A3B or -GGUF repos is titled about the missing LICENSE file (title
filter on the API list: "licen|mit|missing|declar|legal|terms"; discussion bodies not all read). Publisher org name
confusion: Ornith-1.0 was published under `deepreinforce-ai` (LICENSE copyright "deepreinforce.ai"), now `ornith-ai`.

## 2. Base models and inheritance
- Publisher statement (card, blog): "Ornith-1.5 extends Ornith-1.0 (which was developed on top of Qwen3.5 and Gemma4 ...
  continued pretraining, mid-training, and post-training)". Ornith-1.0 README: "post-trained on top of Gemma 4 and Qwen 3.5".
  The 1.5 base itself is not named; architecture is the Qwen3.5-MoE family (qwen3_5_moe, 40 layers, 256 experts), so a
  Qwen weight lineage is near certain, but not stated for 1.5.
- Licenses (HF API cardData, 2026-11-25): Qwen/Qwen3.5-35B-A3B apache-2.0 (license_link .../blob/main/LICENSE), Qwen/Qwen3.6-35B-A3B apache-2.0,
  google/gemma-4-31B apache-2.0 (license_link ai.google.dev/gemma/docs/gemma_4_license; that URL now 301-redirects to
  gemma.ai/resources/apache-2/, which I did not fetch, so the Gemma 4 terms page itself is NOT verified).
- Apache-2.0 allows derivatives under other terms (e.g. MIT) for the new work, but requires that recipients get the Apache
  license text, that modification notices and attribution/NOTICE are kept (section 4). Ornith ships neither a LICENSE nor a
  NOTICE for the Qwen/Gemma part. That is the publisher's compliance gap; it does not take away the user's right to use
  the weights under Apache/MIT (both permissive, commercial use and use without redistribution unaffected). Not checked:
  the actual Qwen3.5 LICENSE text, any Qwen/Gemma use policy beyond the HF tag.

## 3. How others handle it
- HF guidance (https://huggingface.co/docs/hub/model-cards, read today): `license` is metadata "to display the license and
  filter"; "any valid license identifier"; for custom terms use `other` + `license_name` + `license_link`; "If the license is
  not available via a URL you can link to a LICENSE stored in the model repo". HF does not call it a legal grant; it is the
  uploader's declaration, with no check that a file exists.
- Community quantizers keep MIT and the same dead link: bartowski/Ornith-1.5-35B-A3B-GGUF (mit + same link),
  AtomicChat/..-GGUF (mit), mradermacher/Huihui-Ornith-..-abliterated-GGUF (mit). Others relabel: mudler/..-APEX-GGUF and
  mlx-community/..-OptiQ-4bit say apache-2.0 (own choice, base_model ornith-ai). unsloth and lmstudio-community: no Ornith 1.5 35B repo (API error; not found). bartowski's 1.0 GGUF links the 1.0 LICENSE path.
- Ollama: official library entries `ornith` and `ornith-1.5` exist (506K and 807K downloads, "ollama run ornith:35b"); page text says MIT. Ollama blobs/license: nothing retrieved (not checked further). Docker Hub ai/ornith-1.5 exists; no license text read. Sebastian Raschka's gallery lists it. LM Studio and Jan: no catalog entry found (lmstudio.ai/models/ornith = 404); GPT4All and ggml-org lists: not checked. So at least Ollama and Docker (a curated distribution) accepted the same declaration.

## 4. Legal reading (no legal advice)
- A license statement by the rights holder on its official model card, in the GGUF metadata, on the project's GitHub page
  with a full MIT text, repeated across all releases for five months, is an offer of a license by the publisher
  (declaration of intent); a file is evidence, not a precondition. MIT needs only permission plus the notice to be
  included in copies; the GitHub LICENSE gives the notice. Weakness: no LICENSE in the model repos themselves, so
  redistributors (not Tona) cannot easily attach the notice; some uncertainty whether code-repo MIT reaches weights (README
  says it does). Possible rights-holder question: copyright of model weights is itself legally unsettled; MIT as declared is
  the most the user can get.
- Risk for Tona: low. Tona links the user to the publisher's download and does not redistribute. Residual risks: (a) the
  publisher fixes the file later with different terms (unlikely, MIT is declared in 4 places); (b) Qwen/Gemma upstream
  attribution gap is the publisher's problem and the upstream licenses are permissive.

## 5. Recommendation
Offer it. Set `license.name` to "MIT (declared by the publisher)", `verified: true` is NOT warranted by our own rule ("true only
when the license text itself was read") unless we count the GitHub LICENSE as the text; I would set `verified: false` with a
note, or better: verified true with source = the GitHub LICENSE and say in `notes` that the HF repos lack the file. Also
remove "license file missing" from `tona_fit.blockers` (keep the other two blockers, Go and agent eval, which are unrelated).
Add the GitHub URLs to `sources`. Option "wait until the file appears" is not needed and would punish a publisher whose
intent is documented clearly.

Installer line (one line, shown with the model):
- EN: `License: MIT, declared by the publisher (model card, GGUF metadata and GitHub; the LICENSE file is missing in the Hugging Face repository). Built on Qwen/Gemma (Apache-2.0).`
- DE: `Lizenz: MIT, vom Hersteller angegeben (Modellkarte, GGUF-Metadaten und GitHub; die LICENSE-Datei fehlt im Hugging-Face-Repository). Aufbauend auf Qwen/Gemma (Apache-2.0).`
Shorter variant: EN `MIT (publisher's declaration; license file missing on Hugging Face)` / DE `MIT (Angabe des Herstellers; Lizenzdatei fehlt auf Hugging Face)`.
Refinement: say "based on Qwen3.5 / Gemma 4" only as the publisher's statement for 1.0, since the 1.5 base is unnamed.
Housekeeping: re-check the HF repo and GitHub LICENSE before each release (cheap: `GET /api/models/ornith-ai/Ornith-1.5-35B-A3B` siblings and raw LICENSE). Pin the GGUF sha256 as usual; the GGUF metadata already carries `general.license = mit`.

## Checked / not checked
Checked: HF API and raw READMEs of 10+ ornith-ai repos; local GGUF header; GitHub org and repo LICENSE; ornith.ai blog text; HF model-card docs; Qwen/Gemma HF license tags; Ollama library pages; quantizer license tags; discussion titles.
Not checked: Qwen3.5 / Gemma 4 license texts themselves; Gemma terms page (redirected, unfetched); full discussion bodies; LM Studio/Jan/GPT4All/ggml-org lists; Ollama license blob; whether the publisher's entity (deepreinforce.ai vs "Ornith Team") has signed anything else; legal advice.
