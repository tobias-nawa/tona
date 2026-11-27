# Security

## Threat model

`tona-server` has no accounts: anyone who reaches its port and holds the key (or no key is set) can use it.

### Assets

| Asset | Where | Note |
|---|---|---|
| Model files | `~/.local/share/tona/models/`, `work/models/` | public weights; the risk is tampering |
| API key | `~/.config/tona/api-key` | 256-bit hex, mode 0600, by `tona setup` |
| Source code and prompts | request bodies, RAM prompt cache, SSD checkpoints (`TONA_PROMPT_CACHE`: dir 0700, files 0600), `TONA_DUMP_DIR` (every `/v1/messages` body, umask mode, off by default) | a checkpoint is a function of the prompt |
| The machine | GPU, VRAM, RAM | a hung kernel resets the GPU and takes the desktop with it |

### Trust boundaries

| Boundary | Holds | Does not hold |
|---|---|---|
| Network listener | `127.0.0.1:8662`; on loopback a foreign `Host` or `Origin` gets 403 (`TONA_ALLOWED_ORIGINS` adds origins); `0.0.0.0` only if you choose "other PCs, with a key" and a key exists; the server warns at start if it listens beyond loopback without a key | `TONA_USE=server` does not change the bind. No TLS: with "other PCs" key and prompts travel as plain HTTP. `/status` and `/health` need no key and show project, session, agent and role names; `/v1/*` needs it |
| API key | one shared key, constant-time compare, `x-api-key` or `Authorization: Bearer`, 16+ characters; only `/v1/*` checked | key file mode is not checked; no per-user rights |
| Update path | off by default; manifest read only after a minisign signature (prehashed, BLAKE2b-512) verifies against the keys compiled in (`engine/update/keys.h`): the main key signs, the backup key stays offline; lower sequence number or expired manifest refused; size and SHA-256 checked twice; HTTPS only. A compromised release page can withhold updates or offer old signed ones (refused), not an unsigned file. ([docs/use/update.md](docs/use/update.md), [docs/maintain/release.md](docs/maintain/release.md#keys)) | the signing key is the root of trust; if it leaks, installations accept its releases until one signed with the backup key ships |
| Model downloads | `huggingface.co` over HTTPS at a pinned commit, SHA-256 from `toolchain/models.toml` | `tona-server` loads any GGUF path; only installer and `tools/get_model.py` downloads are verified |
| GGUF and request parsing | GGUF loader, API parsers, chat renderer, output parser and update parsers run under libFuzzer with ASan and UBSan (`just fuzz`, [docs/contribute/development.md](docs/contribute/development.md)) | fuzzing proves nothing; a malicious GGUF can attack the server, hence pinned models |
| Chat templates | rendered in C++ per model family; the GGUF's Jinja template is not executed ([docs/how-it-works.md](docs/how-it-works.md)) | other template formats are unsupported, not sandboxed |
| Coding agents | `tona claude`, `opencode`, `junie`, `pi` run with your rights; Claude Code starts in `acceptEdits` (`TONA_MODE`) | no sandbox; the agent and you decide what it may touch |

### Out of scope

- **TLS in the server:** use an SSH tunnel (`ssh -N -L 8662:127.0.0.1:8662 user@host`, client at `http://127.0.0.1:8662`) or
  a TLS reverse proxy in front of the loopback port.
- **Several users on one machine:** one server, one key, one cache. Run one server per user.
- **A hostile local user:** whoever reads your files reads the key and checkpoints.
- **Side channels** (cache-hit timing, GPU and cache contention).

### What counts as a vulnerability

Counts, on a default installation or documented setting: a remote or cross-site request that works without the key or past
the `Host`/`Origin` guard; reading or writing files outside the directories above; memory corruption or code execution
from a request, a pinned GGUF or an update manifest; an update that installs without a valid signature or goes back to an
older release; a secret in a log, a file with a wider mode than stated, or the key in a URL.

Does not count: the out-of-scope items (plain HTTP on a network you opened, another local user, a crash from a GGUF you
wrote), a missing key on a loopback-only server, an agent doing what you allowed, findings without a reproduction.

## Supported versions

Only the latest release.

## Reporting a vulnerability

No public issue. Use GitHub private reporting (**Security and quality** tab, **Report a vulnerability**) or write to
[tobias@tobiasnawa.com](mailto:tobias@tobiasnawa.com). Include the affected release or commit and what an attacker gains.

We need:

- **A reproduction you ran yourself**, minimal, on the latest release or `main`. Reports without one are closed without
  reply.
- **Whether AI tools helped.** Allowed, but you must have verified every claim; fabricated reports are closed without
  reply, repeated ones lead to a block.

You get:

- A first reply within 7 days.
- Coordinated disclosure: fix and advisory public within 90 days by default (sooner if the fix is ready or the flaw is
  exploited), date agreed with you. 90 days is not a promise of a fix.
- A CVE through GitHub ([repository security advisories](https://docs.github.com/en/code-security/security-advisories/working-with-repository-security-advisories/about-repository-security-advisories))
  and credit if you want it.
- Severity judged by this threat model.
