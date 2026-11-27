# Glossary

Terms as this code base uses them. Use exactly these words in code, comments and docs.

| Term | Meaning |
|---|---|
| target model | the large model that produces the output (Qwen3.8-Flash-Next) |
| drafter | proposes several next tokens cheaply: the model's own MTP head or the separate DFlash2 model |
| draft | the tokens a drafter proposes |
| pass | one forward pass of the target model over one or more tokens |
| verify pass | a pass over the drafted tokens; it accepts the longest correct prefix |
| width (of a pass) | number of tokens a pass processes |
| accept / reject | keep or drop drafted tokens after the verify pass |
| rollback | return the recurrent state to the last accepted token (checkpoint or replay) |
| checkpoint | saved copy of state (recurrent layers, KV position) to roll back to or to resume a prompt |
| prompt cache | checkpoints of earlier prompts in RAM and on the SSD, reused when a prompt shares the prefix |
| expert | one feed-forward block of the mixture of experts; the router picks a few per token and layer |
| expert slot | place for one expert in VRAM (the VRAM expert cache) |
| hit / miss | the expert needed by a pass is / is not in a VRAM slot |
| pinned | page-locked host RAM the GPU reads directly over PCIe |
| page cache | the kernel's file cache; model files not pinned are read through it |
| layer, head, row, column | as in the model: transformer layer, attention head, matrix row and column |
| hyper-connection (HC) | the model's residual mixing across several streams (`gr.down`, `gr.up`) |
| QSA | the model's sparse attention layers with an indexer |
| DeltaNet | the model's linear-attention (recurrent) layers |
| PLE | per-layer embeddings: the n-gram table (28.8 GB) read per token |
| OEA | overlap of experts across parallel sequences: share experts both need in one pass |
| sequence | one conversation the server is serving (`seq`, `TONA_SEQS`); "conversation" in user-facing text |
| megakernel | the persistent GPU kernel that runs a whole pass as a program of ops |
| op | one step of the megakernel program (a matvec, a norm, a wait for the CPU …) |
| block, wave, lane | GPU workgroup, wavefront (32 lanes on RDNA4), single lane |
| barrier | in-kernel synchronization of all blocks |
| abort | ending a pass early after a timeout (error) |
| yield | pausing the megakernel while it waits for the CPU, resumed by the host (no error) |
| effort | thinking budget per request (`low`, `medium`, `high`, `xhigh`, `max`, as far as the chat template knows them) |
| ceiling | the speed limit the hardware allows, computed from measured bandwidth |
| reference platform / backend | the machine and backend the engine is optimized for (docs/backends/reference-platform.md) |
| decode / prefill | generating tokens / processing the prompt; always say which one a rate is |
