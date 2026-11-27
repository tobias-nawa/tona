# Memory for further conversations: give up expert slots or keep the state in RAM (stage 3)

Own measurement, 2026-10-26, one conversation, bench-set (greedy, MTP), one server start per variant with two runs.
State per conversation: 149 MB (measurements/2026-10-26-overlap and the log of saved conversations).

| Variant | Expert slots per layer | Code tok/s | Text tok/s |
|---|---:|---:|---:|
| base | 85 | 90.9 / 90.3 | 65.5 / 63.8 |
| a) room for 2 conversations (IE_VRAM_RESERVE_MB=790) | 83 | 90.6 / 90.6 | 65.2 / 64.2 |
| a) room for 4 conversations (IE_VRAM_RESERVE_MB=1090) | 80 | 90.1 / 90.2 | 64.9 / 63.9 |
| b) state in pinned RAM (IE_STATE_HOST=1) | 87 | **83.9 / 84.0** | **59.6 / 58.7** |

- a) costs nothing measurable: 2–5 slots fewer per layer change decode by at most 0.5 % (within the noise of the
  repetition). The expert cache has hardly any marginal benefit left at 80–85 slots.
- b) costs 7.5–8 % decode: the state is read and written over PCIe on every pass.
- Base with 85 instead of 88 slots: the desktop used more VRAM that day.

Conclusion: allocate state slots for further conversations in VRAM at start (no borrowing at runtime needed).
Default 2 conversations, up to 4 with the same result.
