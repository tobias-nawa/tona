# 42: Licenses of the models we derive and redistribute

Research of 2026-11-11 for publishing our own requantized models (medium tier, #108) on Hugging Face and fetching them
with `tona-install`. Not legal advice. [V] = license text or model card read directly (raw files on Hugging Face,
retrieved 2026-11-11), [S] = secondary source.

## Base model: Qwen3.8-Flash-Next, Qwen Community License 1.0 [V]

Source: `huggingface.co/Qwen/Qwen3.8-Flash-Next`, `README.md` (`license: other`, `license_name: qwen-community-1.0`) and
`LICENSE` (Copyright (c) 2026 Qwen).

- Grants use, copy, modify, merge, publish, distribute, sublicense, sell, deploy, host, fine-tune and derivative works,
  weights included.
- Condition 1: the copyright and permission notice go with all copies or substantial portions (ship `LICENSE`). A
  commercial product or service of the licensee with more than 100 million monthly active users or more than
  USD 20 million monthly revenue must show the model name prominently; this also applies to derivatives.
- Condition 2: a "Model as a Service" or "AI Work Assistant" business needs a separate license from Qwen before any
  commercial use. Purely internal use that exposes nothing to third parties is exempt. "AI Work Assistant" means a
  standalone AI product mainly for AI-assisted coding or office productivity; single-purpose tools and AI features in
  products with another main purpose are exempt.
- Not in the text: no "Qwen" in the derivative's name, no "Built with Qwen" (that was the older Qwen 2024 license [S]),
  no copyleft, no separate acceptable use policy.

Ambiguous: whether Tona with its agent launchers counts as an "AI Work Assistant" once it is offered commercially (as
a free MIT project without revenue the condition does not bite); self-hosting by users is no "Model as a Service" by
the wording.

## Sources of our derivatives [V]

- **ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF:** YAML says `license: apache-2.0`, but the card's License
  section says "These weights inherit the license of the base model". No `LICENSE` file in the repository. ISTA cannot
  grant more than Qwen granted, so we treat the Coder and everything copied from it as Qwen Community License material.
- **unsloth/Qwen3.8-Flash-Next-GGUF:** `license: other`, `license_name: qwen-community-1.0`; no `LICENSE` file.

## Small-tier candidates [V]

| Model | License | Base | Duties for requants |
|---|---|---|---|
| Ornith-1.5-35B-A3B | MIT (YAML; the linked `LICENSE` file is missing) | card names Qwen3.5 and Gemma 4 for Ornith 1.0, the base of 1.5-35B-A3B is not stated; both are Apache-2.0 | MIT notice; Apache-2.0 duties of the base if it is Qwen3.5 (license copy, change note, NOTICE) |
| Groot-19B-A3B | Apache-2.0 (linked `LICENSE` file missing) | Qwen3.6-35B-A3B, Apache-2.0 | Apache-2.0 license copy, change note, copyright notices |

Neither falls under the Qwen Community License 1.0.

## Checklist for publishing a Qwen3.8-Flash-Next derivative

1. Card YAML: `license: other`, `license_name: qwen-community-1.0`, `license_link: LICENSE`,
   `base_model: Qwen/Qwen3.8-Flash-Next`, `base_model_relation: quantized`.
2. Verbatim copy of the Qwen `LICENSE` in the repository.
3. Card: origin (base model, ISTA expert selection with links and its papers, our changes: experts removed, requantized
   with our imatrix, which tensors are byte copies), a note on condition 2 for users, and a note that the ISTA card's
   Apache tag contradicts its own text.
4. A name that makes the base clear without looking like an official Qwen model.
5. `tona-install`: show license name and link before the download; a confirmation for Qwen3.8 models is not required
   by the text but advisable because of condition 2.
6. Tona stays MIT; README and installer say that each model has its own license.

## Open

- AI Work Assistant question for a commercial offer: ask Qwen (model-business@notice.qwencloud.com).
- The contradicting license tag at ISTA is unresolved.
- Base of Ornith-1.5-35B-A3B; missing `LICENSE` files at Ornith and Groot.
