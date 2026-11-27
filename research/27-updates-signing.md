# Updates and release signing (2026-10-28)

Agent research (two agents, web) plus own measurements on this machine (section "Own measurements").

## Own measurements (2026-10-28)

- A clean build of the engine (`cmake` + `ninja -j12`, GPU part included) takes 88-89 s on the reference machine.
- `tona-server` links the ROCm runtime (`libamdhip64`, HSA, comgr) from the toolchain wheels through an absolute
  `RUNPATH` into the checkout, plus the system `libstdc++` and `libgomp`. Both a source and a binary update need the
  pinned ROCm runtime on the machine; binaries would need a relative `RUNPATH` (`$ORIGIN`).
- `tona-top` embeds the absolute repository path (`TONA_REPO_DIR`).
- Reproducibility: two clean builds in different build directories differed in all 23 executables. With
  `-ffile-prefix-map=<build dir>=build` 16 of 23 matched; the 7 with device code differed in the HIP compilation-unit
  id (`__hip_cuid_*`), which clang derives from the output path. With a fixed `-cuid` per source (hash of its path in
  the tree, now in `engine/CMakeLists.txt`) all 23 executables are bit-identical, and `scripts/isa-diff.py` reports all
  171 kernels unchanged. Not yet tested: different checkout paths (source prefix map), another machine, a container.

## Part 1: update channels, signing, TUI updater


Legend: [V] primary page read in this run, [S] search hit only, [K] my own background knowledge, not verified now, [?] could not verify.
Web search in this environment was weak (several queries returned noise); every [K]/[?] item should be checked before it goes into a design.

### 0. What the repo does today (read from the repository)

- Users **build from source**. No release binaries, no packages. `scripts/setup.sh` makes a venv `work/rocm-10.1`, installs ROCm 10.1 wheels from `https://stable.repo.amd.com/rocm/whl-next/` with `pip --require-hashes -r toolchain/rocm-10.1.lock` (hash-pinned, no sudo), runs a WMMA smoke test, then `source toolchain/env.sh`. The engine is built with CMake/Ninja + hipcc (`engine/CMakeLists.txt`: needs ROCM_PATH and TONA_HIPCXX, else the GPU part is skipped). Everything lives in `work/` (git-ignored). `tona-install` (C++/FTXUI, `installer/`) detects hardware, checks packages (only prints the install command, e.g. `curl -fsSL https://claude.ai/install.sh | bash`), downloads models, does a trial run via `just start`.
- The server is `work/build/engine/tona-server`, started by `scripts/tona-service.sh` as systemd **user** unit `tona-server` (`~/.config/systemd/user/tona-server.service`, ExecStart has an absolute path into the checkout, `work/build/...`). `tona-top` (engine/tools/tona_top*.cpp) polls `/status`, key `s` runs `TONA_START_CMD` or `just start`, `x` stops. It has no network access beyond the server URL.
- Models: `toolchain/models.toml` pins HF `repo` + full commit `revision` + per file `size` + `sha256`; `tools/get_model.py` downloads `.part` with HTTP Range resume, verifies SHA-256, caches a `.sha256ok` marker. A comment documents that ISTA-DASLab replaced a shard in place on 2026-10-19 on `main`, so the pin on purpose points to the 2026-09-29 revision. This is the strongest argument for pinned revisions.
- Independent version axes (today all tied to the git checkout, none has its own number):
  1. engine binaries (tona-server, tona-top, tona-gpu-gen), including GPU code built by hipcc at build time (so gfx1200 code objects depend on the compiler version);
  2. ROCm toolchain (`rocm-10.1.lock`, ~7 GB, setup.sh requires 12 GB free);
  3. models + drafters (`models.toml`, ~55 GB, plus expert profiles in `toolchain/profiles`);
  4. installer `tona-install` (own CMake project, FTXUI fetched via FetchContent);
  5. config/state outside the repo: `~/.config/tona/api-key`, `work/tona.conf`, `work/prompt-cache`, `work/profile/...`.
- `.github/workflows/check.yml` exists (CI only). No release workflow. `SECURITY.md` exists.
- Consequence: an updater first needs a decision "source or binary". A prebuilt gfx1200 binary is possible because the reference platform is a single target, but the binary's dependence on the ROCm runtime (rpath into `work/rocm-10.1`?) is **[?] not checked**; the CMake lines suggest it links against the venv ROCm.

### 1. Update channels

### 1.1 GitHub Releases
- Unauthenticated REST limit: **60 requests/hour per IP** [V https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api, read 2026-10-28]. The page says nothing about release-asset downloads or ETag/304 handling [V: not stated]. From [K]: `github.com/<o>/<r>/releases/download/<tag>/<asset>` and `.../releases/latest/download/<asset>` are plain CDN redirects, not API calls, so a signed `manifest.json` fetched from `releases/latest/download/` avoids the API and its limit. Mark [K, test it].
- Since 2025-06-03 the API/GraphQL/`gh release view` expose a **SHA-256 `digest` per release asset** [V https://github.blog/changelog/2025-06-03-releases-now-expose-digests-for-release-assets/]. Useful as a second integrity source, but it is generated by GitHub, so it proves nothing against a hijacked account.
- **Immutable releases**: tag and assets cannot be changed or deleted after publishing, tag names cannot be reused even after repo delete/recreate, and a release attestation (tag, commit SHA, assets) is created automatically; recommended flow: draft, attach all assets, publish [V https://docs.github.com/en/code-security/supply-chain-security/understanding-your-software-supply-chain/immutable-releases]. GA date not stated on the page [?]. Worth enabling: it stops "silent asset swap" on an existing version.
- Release assets vs the automatic source tarball: auto-generated source archives are not stable byte-for-byte guarantees across time [K, historic 2023 checksum-change incident]; upload your own `tona-<ver>-src.tar.zst` and binary tarballs as assets and sign those.

### 1.2 Git tag + rebuild
This is today's flow. Pros: nothing to sign beyond git tags (`git tag -s`/SSH-signed tags [K]); cons: needs compilers, 7 GB ROCm, ~minutes of build, GPU code objects rebuilt. Fine for developers, poor for an "update" button.

### 1.3 Distro packages, Flatpak, containers
- **Flatpak**: the Flatpak extension docs list only graphics (GL), Vulkan, GStreamer, VAAPI, icon themes; they do not mention ROCm or `/dev/kfd` [V https://docs.flatpak.org/en/latest/extension.html]. A ROCm runtime extension would have to be built by us. Not sensible for a GPU compute server with a systemd user unit. [S/K]
- **COPR/RPM**: workable for Fedora (one reference distro), signed by the COPR GPG key, and `dnf` handles update/rollback. But the toolchain is an AMD pip wheel set outside the RPM world (system ROCm 7.1.1 is explicitly not used per CLAUDE.md), so an RPM would either bundle 7 GB or depend on a second step. Good later as a thin RPM that ships tona binaries + unit file. [K, not researched deeper]
- **Containers**: vLLM ships `vllm/vllm-openai-rocm` on Docker Hub (AMD's older `rocm/vllm` deprecated) [S web search 2026-10-28, docs.vllm.ai/en/stable/deployment/docker.html]; llama.cpp ships `ghcr.io/ggml-org/llama.cpp:server-rocm` [S rocm.docs.amd.com llama.cpp page]. Fits servers, not a desktop user who wants the TUI, systemd user service and sysfs VRAM accounting; needs `--device=/dev/kfd --device=/dev/dri`. Optional extra channel, not primary.

### 1.4 What comparable projects do
| Project | Mechanism | Verification | Source |
|---|---|---|---|
| Ollama (Linux) | no updater; re-run `install.sh`, optional `OLLAMA_VERSION`; downloads `ollama-linux-<arch>.tar.zst` (+ `-rocm` archive), writes a system-level unit, restarts it | **none** besides HTTPS | [V https://github.com/ollama/ollama/blob/main/scripts/install.sh] |
| Ollama (Windows) | built-in updater; CVE-2026-42248: update executables are not integrity/authenticity checked | none | [S https://osv.dev/vulnerability/CVE-2026-42248] |
| llama.cpp | GitHub Releases with prebuilt zips (Linux mostly CPU/Vulkan; ROCm via container or community builds), people rebuild from source | none in-band known [?] | [S rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/llama-cpp-install.html] |
| LocalAI | `curl https://localai.io/install.sh \| sh`, binary download, Docker recommended | [?] | [S https://localai.io/docs/advanced/installer] |
| KoboldCpp | single binary per release on GitHub, manual download | [?] | [S newreleases.io koboldcpp v1.117.1] |
| LM Studio, Jan | in-app updaters (Electron/Tauri style) | [?] could not verify; do not cite | - |
| gh | no self-update; checks for new version at most every 24 h on any command, prints notice to stderr; `GH_NO_UPDATE_NOTIFIER` disables | distro/brew handle it | [V https://cli.github.com/manual/gh_help_environment] |
| rustup | `rustup self update`; `auto-self-update = disable \| enable \| check-only`; channels stable/beta/nightly | not covered on the page | [V https://rust-lang.github.io/rustup/basics.html] |
| deno | `deno upgrade` from GitHub releases; channels stable/lts/rc/canary; `--checksum=<sha256>`, `.sha256sum` published; delta patches (2.8+) verified against SHA-256 and fall back to full archive; `--dry-run`; archive cache; rollback only by installing an older version | checksum only | [V https://docs.deno.com/runtime/reference/cli/upgrade/] |
| uv | `uv self update` only for the standalone installer; other installs defer to their package manager; based on cargo-dist/axoupdater with install receipts (GitHub Releases) | [?] not read | [S https://docs.astral.sh/uv/ , https://docs.rs/axoupdater] |
| mise | `mise self-update` [K, not researched]; says minisign-style verification? [?] | - | - |
| Rust `self_update` crate (used by many CLIs) | backends GitHub/GitLab/S3/static manifest.json; flow: checksum, then **signature (zipsign, ed25519ph)**, extract, verify, replace; docs say the digest is "an integrity check only, not a substitute for signatures" | | [V https://docs.rs/self_update/latest/self_update/] |

Pattern: the popular LLM tools update by "rerun installer" and mostly skip signatures; the tools with a real updater (deno, self_update users) use a checksum, a few add signatures. Doing signatures is therefore above the field, not an exotic choice.

### 2. Integrity and authenticity

Checksum alone only detects corruption if it is fetched from the same place as the artifact (same attacker). It needs a signature rooted outside GitHub's storage.

| Scheme | Fits a C++ TUF-less updater? | Notes |
|---|---|---|
| **minisign / signify** (Ed25519, ISC) | **Best fit.** Reference tool is C (needs libsodium for CMake build) [V https://github.com/jedisct1/minisign]; verifier libraries exist in Go, Rust, Python, .NET, Elixir, and a C library "libminisign" by a third party (kroket.io, 2025-09) [S https://kroket.io/blog/libminisign.html]. Format is tiny; verification is Ed25519 verify (plus BLAKE2b for the prehashed "ED" mode) so it can be embedded with libsodium or Monocypher [K]. Trusted comments are signed, so version/timestamp can be bound into the signature. Upstream still active (0.12 packaged early 2026) [S packages.debian.org/sid/misc/minisign] |
| **cosign keyless + Sigstore bundle** | Verification needs: bundle (cert chain + signature + Rekor inclusion proof/SET), a trusted root (TUF-distributed), X.509 + certificate-identity checks, Merkle proof, SCT. Offline verification is supported by the bundle format and `cosign verify-blob --bundle ... --certificate-identity ... --certificate-oidc-issuer ...` [V https://docs.sigstore.dev/cosign/verifying/verify/ , https://docs.sigstore.dev/about/bundle/]. **I found no C/C++ verifier** (search for sigstore-cpp returned nothing; clients are Go, Python, Rust, Java, JS, Ruby; sigstore-rs calls itself experimental, pre-1.0 [V https://github.com/sigstore/sigstore-rs]). So in C++ you shell out to `cosign` or `gh`, which adds a 100+ MB Go dependency the user may not have |
| **GitHub artifact attestations (SLSA provenance)** | CI step `actions/attest@v4` with `id-token: write, attestations: write` [V https://docs.github.com/en/actions/security-for-github-actions/using-artifact-attestations/using-artifact-attestations-to-establish-provenance-for-builds]. Verify with `gh attestation verify FILE -R owner/repo`; offline with `--bundle` and `--custom-trusted-root`; pin the signing workflow with `--signer-workflow`; with `--bundle` the repo name is only compared to the certificate and a renamed/transferred repo is not detected [V https://cli.github.com/manual/gh_attestation_verify]. Proves "built from this repo by this workflow at this commit", not "the maintainer approved it". SLSA level claim not on the page [?] |
| **GPG** | Works, heavy to embed (needs gpg/gpgme), key UX poor, even CPython deprecated PGP signatures in favour of Sigstore (PEP 761) [S discuss.python.org, 2024]. Only sensible for distro/COPR repos |
| **TUF** | Provides roles, thresholds, expiry, snapshot/timestamp (defends rollback, freeze, mix-and-match, key compromise with threshold). python-tuf is the reference implementation, "version 1.0 of the TUF spec" [V https://github.com/theupdateframework/python-tuf]; the page lists no C/C++ client. go-tuf, tuf-rs exist [K]; a C++ client [?] none known. For one maintainer and four artifact types it is **over-sized**; but its two cheap ideas are worth copying (below) |

Rollback/freeze defence without TUF: sign a small `manifest.json` containing `version` (monotonic integer `seq`), `expires` (e.g. 30 days), artifact names, sizes, SHA-256s, required toolchain-lock hash, min-installer version. The client stores the highest `seq` it ever accepted and rejects lower; rejects an expired manifest as "stale, mirror may be frozen" (warn, not brick). This gives the bulk of TUF's timestamp+snapshot guarantees. [K, design]

Key management and compromise:
- Signing key compromised: attacker can sign malicious updates until clients know. Mitigation: embed **two** public keys in the installer (active + offline backup); rotation message = new manifest signed by the old key adding the new key; revocation of a leaked key needs an out-of-band step (README/website fingerprint, reinstall) because clients cannot learn about it from a channel the attacker also controls. Keep the secret key on a hardware token or an offline machine, passphrase-protected (minisign encrypts keys) [V minisign README].
- GitHub account/CI compromised: attacker can publish assets and CI-held secrets. If the minisign key is only on the maintainer machine, a hijacked account cannot produce valid signatures; attestations would show the compromised workflow identity, so they are *not* a defence here. With keyless Sigstore the identity is "the workflow", so anyone controlling the repo controls the signer. Therefore: **two independent roots**: attestation (CI built it) + minisign (maintainer approved it).
- Trade-off: a manual signing step per release; acceptable at a release cadence of weeks.

### 3. Models

- Pinning: `revision` accepts a full 40-char commit hash (short hashes are not accepted) [V https://huggingface.co/docs/huggingface_hub/guides/download]. Current repo already pins revision + SHA-256 + size, which is the right design. HF `main` can change under you (own evidence, models.toml comment, 2026-10-19).
- Source of truth for hashes: `GET /api/models/{repo}/tree/{revision}` returns `lfs.oid` (SHA-256) and `size` (from installer research 22, [V there]). The sha256 in models.toml should be signed as part of the release (it is, via the repo/manifest), not trusted from HF at update time; HF gives transport, we give trust.
- Resume: plain `resolve/<revision>/<path>` with `Range:` works (already implemented in `get_model.py`). The official client uses Xet storage since huggingface_hub 0.32 (`hf_xet`), `hf_transfer` deprecated [V same guide]; community reports of stalled/zero-exit incomplete Xet downloads with workaround `HF_HUB_DISABLE_XET=1` [S discuss.huggingface.co/t/download-resume-possibility/180468, anecdotal]. Reason to keep own HTTP downloader: deterministic resume, no Python dependency in the C++ installer.
- Updates independent of engine: ship `models.toml` as part of the signed manifest (or as `models-<n>.toml`, signed separately) so "new model/revision available" is a model-channel update, never forced; engine releases declare only a format/min-version of GGUF support. New revision = new directory by (name, revision[:12]); old one deleted only after the new one verifies and the user confirms (55 GB each; check free space first, `df`).
- Disk: write into a non-tmpfs location under the user's data directory, never `/tmp` (tmpfs on many systems).

### 4. TUI updater UX (tona-top)

Principles: opt-in; read-only by default; nothing leaves the machine except the update fetch itself.
- Check: off by default; first run of the TUI asks once ("check GitHub for new versions weekly? y/N"), setting in `~/.config/tona/config` like rustup `auto-self-update = disable|check-only|enable` and gh's 24 h throttle + env kill-switch (precedents: [V] above). Manual `u` key always works. The request carries no identifiers: GET of one static manifest URL (User-Agent `tona/<version>` only; GitHub still sees the IP; state this in docs). No telemetry, no install ID.
- Show: banner "0.9.2 -> 0.10.0 available", `Enter` shows changelog (release notes text embedded in the signed manifest, rendered in a panel), size of download, toolchain change yes/no, model change yes/no, "restart needed".
- Download: background thread (tona-top already has `in_background`), `.part` + Range resume, progress gauge, SHA-256 + minisign check before anything is installed. Server keeps serving during download.
- Install: versioned dirs `~/.local/share/tona/versions/<ver>/`, shared `toolchains/<lock-hash>/`, `models/<name>-<rev>/`, `current -> versions/<ver>` swapped with `symlink(tmp); rename(tmp, current)` (atomic [K POSIX]). Unit file uses `ExecStart=%h/.local/share/tona/current/bin/tona-server` so the symlink is all that changes. Keep N-1 (and N-2 if disk allows).
- Restart: wait until no running requests (tona-top already sees `/status`), then `systemctl --user restart tona-server`; poll `/status` until ready (model load takes time: pinning ~25 GB RAM); if not healthy within a timeout, **auto-rollback** (swap symlink back, restart) and report. Never restart while a drain is refused without the user's confirmation. Beware the GPU-hang constraint from the project memory: do not run blocking HIP calls in the updater.
- Rollback: key `r` or `tona-install rollback` = swap to previous version dir; since dirs are immutable, it is instant and offline. Anti-rollback seq (section 2) must allow a user-initiated rollback to a *previously installed* version (store accepted-seq separately from installed version).
- GPU kernels / toolchain: manifest field `toolchain_lock_sha256`. If equal to installed: binary-only update. If different: stage new toolchain into `toolchains/<hash>/` (7 GB, hash-verified by `pip --require-hashes`, resumable via pip cache), run the smoke test (`smoke_wmma`) *before* switching; if the release ships prebuilt gfx1200 binaries they must be built against exactly that ROCm (code objects + runtime ABI) - rebuild only needed for source installs. Source-build users: updater does `git fetch --verify-signature` of a signed tag, then `just build` into a new dir.
- Self-update of the installer/TUI: it replaces itself last (write new file, `rename` over, re-exec), like deno/rustup do for themselves [V]; a failure leaves the old binary.

### 5. Recommendation

**Channels (in order):** (1) GitHub Releases as the single source of truth: signed source tarball + signed prebuilt `tona-<ver>-linux-x86_64-gfx1200.tar.zst` (tona-server, tona-top, tona-install, unit file template) + `manifest.json` + `manifest.json.minisig`; (2) git tag build for developers; (3) later a thin COPR RPM; containers optional. No Flatpak.

**Signing:** minisign Ed25519 as the trust root (maintainer key offline, two public keys embedded, rotation via old-key signature), **plus** GitHub artifact attestations for provenance and immutable releases switched on. Client verifies *only* minisign natively in C++ (embed ~300 lines or Monocypher/libsodium; Ed25519 + BLAKE2b-prehash; test vectors from minisign) - no cosign/gh needed. Attestation verification is an optional "paranoid" step that shells out to `gh attestation verify --bundle ... --signer-workflow owner/repo/.github/workflows/release.yml` if `gh` exists; documented for users, not required. Rejected: GPG (UX), keyless-only (no C++ verifier, identity = whoever controls the repo), full TUF (no C++ client known, over-sized) - but copy TUF's monotonic seq + expiry.

**Release workflow (GitHub Actions, sketch):** tag `v*` push -> job builds in a clean Fedora container with the pinned ROCm lock (hash-checked), runs tests + smoke where a GPU is available (self-hosted runner is [?]; otherwise the maintainer runs BENCHMARKS.md and the gate before signing), produces tarballs + `SHA256SUMS`, `actions/attest@v4` for each tarball, uploads to a **draft** release; maintainer verifies attestation (`gh attestation verify`), builds `manifest.json` (seq, expires, sizes, sha256, toolchain hash, models.toml hash, changelog), signs offline with minisign, uploads, publishes (becomes immutable). Pin all Actions by commit SHA [K best practice].

**Rollback:** versioned dirs + symlink + automatic health-check rollback + manual `r`.

**Trade-offs:** (a) prebuilt binaries tie to the ROCm 10.1 layout and gfx1200 only (fits the reference-platform policy; other GPUs stay source builds); (b) offline-key signing adds manual steps and a bus-factor risk (keep an encrypted backup of the key, document recovery); (c) opt-in checks mean many users run old versions - acceptable, security fixes are flagged in the manifest (`severity`) and shown at the next TUI start; (d) GitHub is a single transport; mirror manifest on a second host later, signatures make mirrors safe; (e) minisign key leak = rotate through old key; if both keys leak, users must reinstall manually.

### Not verified / open
- Real update mechanisms of LM Studio, Jan, mise, llama.cpp checksums; uv's exact verification; rustup's hash/signature verification.
- Whether `releases/latest/download/` redirects count toward the 60/h limit and ETag/304 behaviour on the API.
- GA date and exact semantics of immutable releases; SLSA level of GitHub attestations.
- Any C/C++ Sigstore or TUF client (none found, absence is not proof).
- Whether the built tona-server is relocatable w.r.t. the ROCm runtime (rpath/venv paths) - needs a `readelf -d` check before promising prebuilt binaries.

## Part 2: security in depth, keys and maintainers


Legend: [V] primary page fetched this run; [S] search-result summary only (secondary, check before relying); [K] my background knowledge, not verified now; [E] own experiment on this machine; [?] unverified.
Web tooling here returned summaries, not raw pages, so most "[S]" items are second-hand. Items that gate the design are listed in "Open / verify first" at the end.

### 0. Decision

**C (hybrid), staged.** CI builds binaries unprivileged and attests them. The maintainer rebuilds locally in the same pinned environment, compares hashes, and only then signs the manifest with an offline hardware-backed key. Source rebuild stays the documented fallback and the only path for non-gfx1200 GPUs.
Why not A alone: it moves no trust problem away. The user's machine runs the same compilers plus a huge build, and the source tarball is the xz attack surface. A is, however, the correct fallback and the thing C's "independent verification" is checked against.
Why not B alone: the signed bits come out of a CI that can be poisoned (Ultralytics, tj-actions, Trivy below) and nobody checked them. B's only answer is "trust CI".
C needs reproducibility; section 3 shows it is plausible but unproven for the full engine (open item 1).

### 1. Threat model, per attack

Attack names are the TUF spec's [V https://theupdateframework.github.io/specification/latest/, fetched 2026-10-28]: arbitrary installation, endless data, extraneous dependencies, fast-forward, indefinite freeze, malicious mirrors, mix-and-match, rollback, key compromise, wrong software installation. "Slow retrieval" is not in the current spec text as far as the fetch showed [V]; treat it as a client timeout problem.

Legend of columns: M = minisign-/SSHSIG-signed manifest with `seq`, `expires`, per-file SHA-256, size; I = GitHub immutable releases; T = GitHub artifact attestations.

| Attack | Covered by | What remains |
|---|---|---|
| Arbitrary install / wrong software / tampered CDN | M (client checks sig then SHA-256 and size). I,T add nothing the client relies on | nothing, if client verifies before use |
| Rollback | M with client-stored highest `seq` | user-initiated rollback must be a separate, local path (accepted-seq vs installed version) |
| Indefinite freeze | M `expires` (client warns "stale" after expiry) | an attacker who blocks only the manifest URL cannot be told from an outage; show the age in the TUI |
| Mix-and-match | M if the manifest lists every artifact of the release (single signed file) | only one metadata file => no mix-and-match by construction. Do not sign artifacts separately |
| Fast-forward (attacker inflates version) | M: signature needed, so only after key compromise | after key rotation, root metadata must reset or allow `seq` reset; follow TUF (clients drop trusted timestamp/snapshot on key rotation) [V] |
| Endless data / slow retrieval | client: byte caps from the manifest size, libcurl timeouts, low-speed limit | M must be small (cap e.g. 256 KiB before parse) |
| Compromised CDN / GitHub storage | M | availability only |
| Compromised **maintainer GitHub account** | M if key is off GitHub; I only prevents altering an already-published version, **not** publishing a new one; T would show the attacker's own workflow as valid | attacker can DoS (delete repo/unpublish) and push poisoned source; the client must therefore never trust "latest" without signature |
| Compromised **CI** (workflow injection, cache poisoning, malicious action) | **C only**: signing key is not in CI and the maintainer's independent rebuild must match. M alone (design B) signs whatever CI produced | reproducibility must hold; if rebuild differs, release stops |
| Compromised build dependency (ROCm wheel, compiler, apt/dnf pkg) | hash-pinned wheels (`pip --require-hashes`) stop silent change after the lock; a bad wheel at lock time is not detected by anything here | the lock hashes are the trust anchor; AMD's own build/signing is out of scope. Record toolchain lock hash in manifest |
| Compromised signing key | threshold (section 8) + offline hardware key + rotation | users who miss the rotation window need manual trust reset |
| Malicious mirror | M | none |
| Compromised **user machine** | nothing | out of scope |
| xz-style: release artifact differs from git | C: source tarball = `git archive` of the signed tag, rebuilt by the maintainer and by a verifier; reproducible binary hash must match | nothing automatic checks git vs tarball for third-party deps; vendor nothing |

Immutable releases [V https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases]: tag locked to a commit, assets cannot be modified or deleted, a release attestation (tag, commit SHA, assets) is created automatically, deleted immutable release's tag name cannot be reused, repo delete+recreate does not free the name; title/notes remain editable (so do not put security-relevant text only in the notes; put it in the signed manifest); assets must be attached before publishing (draft first). The page says nothing on auto-generated source archives [V] => never use them. GA announcement dated 2025-10-28 [S https://github.blog/changelog/2025-10-28-immutable-releases-are-now-generally-available/]. Only tags tied to published releases are protected, plain tags are not [S korben.info / community discussion 178351]; add a tag ruleset (section 7).
Attestations prove "this workflow in this repo built this at this commit" [V earlier report: cli.github.com/manual/gh_attestation_verify]; they do not prove a person approved. Under a stolen account they are worthless as authenticity, useful as provenance for the maintainer's own pre-sign check.

### 2. Incidents 2024-2026 and lessons

| Incident | Facts | Lesson for us |
|---|---|---|
| **xz-utils** (CVE-2024-3094, 2024-03) | Malicious `build-to-host.m4` was in the upstream release tarballs but not in git; GitHub auto-generated archives did not contain it; tarballs carry generated files that differ from the checkout [S oss-security 2024/03/29/8 (Andres Freund), arXiv 2404.08987, Wikipedia, linuxiac] | Release tarball must equal `git archive` output of the tag (no generated files, no autotools output). Verify in CI and locally: `git archive` hash == tarball hash. Sign the tarball hash in the manifest. Binary built from that tarball |
| **tj-actions/changed-files** (CVE-2025-30066, 2025-03-14/15) | attacker repointed version tags to a malicious commit, secrets dumped to logs, >23,000 repos used it [S wiz, orca, sentinelone, GHSA-mrrh-fwg8-r2c3] | Pin every action by full commit SHA; never give the build job release secrets; restrict `GITHUB_TOKEN` to `contents: read`; do not echo secrets |
| **Ultralytics PyPI** (2024-12-04..) | script injection via branch names in a workflow; malicious 8.3.41/8.3.42, later 8.3.45/46 published straight to PyPI (token stolen or account control) [S hiddenlayer, wiz, bleepingcomputer]. **The "cache poisoning" claim in the brief is not supported by the sources I found; it is script injection (plus later direct upload)** | no `${{ github.head_ref }}` / untrusted fields in `run:`; build only on tag push, never `pull_request_target`; the upload credential (here: signing key) must not be reachable from a workflow |
| **Trivy / trivy-action** (2026-03-19, CVE-2026-33634 per one source) | 75/76 (or 76/77) tags repointed to a credential stealer; root cause: incomplete credential rotation after an earlier breach (2026-02-28); safe pins were SHAs [S wiz, phoenix.security, snyk, penligent; counts differ between sources] | after any suspected incident rotate *everything*; SHA pins; also relevant: security tooling in CI is part of the attack surface |
| **Notepad++ updater** (June-Dec 2025, disclosed 2026-02) | hosting provider breached; traffic to update XML redirected for selected users; updater (WinGUp) did not verify installer signature before 8.8.9 (CVE-2025-15556); 8.9.2 added XMLDSig on the *manifest* ("double lock") [S theregister 2026-02-02, thehackernews, securityaffairs, ostechnix, advancedinstaller] | exactly our model: sign the manifest *and* hash the files; HTTPS alone does not help against a compromised host; target-specific redirection is invisible to everyone else, so a transparency/monitoring layer (section 9) is the only detection |
| Ollama Windows updater (CVE-2026-42248) | update executables not integrity/authenticity checked [S osv.dev, from earlier report] | field default is "no verification"; we exceed it |
| Others I did not research [?] | npm "Shai-Hulud" (2025), nx "s1ngularity" (2025), codecov bash uploader (2021), SolarWinds | not cited; do not use without checking |

Common thread: tags are mutable, CI secrets leak, an unsigned or CI-signed artifact is only as safe as the weakest workflow. All three CI incidents are defeated for *us* by: key not in CI + independent rebuild + SHA pins + signed manifest.

### 3. Reproducible builds for this stack

Known-good flags [V https://reproducible-builds.org/docs/build-path/, fetched 2026-10-28]: `-ffile-prefix-map=OLD=NEW` (alias of `-fdebug-prefix-map` + `-fmacro-prefix-map`, GCC 8 / Clang 10+); fixed build path is not always enough because of randomly named intermediates; dpkg `fixfilepath` exists. `SOURCE_DATE_EPOCH`: GCC >= 7 and clang >= 16 support it [V https://reproducible-builds.org/docs/source-date-epoch/]; the page does not explain `__DATE__`/`__TIME__` handling, so: **do not use `__DATE__/__TIME__/__TIMESTAMP__` at all** (and add `-Wdate-time -Werror`) [K].
Other flags [K]: `-frandom-seed=<constant or hash of file name>` (GCC; anonymous-namespace symbol names), `-Wl,--build-id=sha1` (content-derived, reproducible) with `-Wl,--sort-common`/`--hash-style=gnu` not needed, `-fno-record-gcc-switches`, `ar D` / `-Wl,-O1` irrelevant, `LC_ALL=C`, `TZ=UTC`, `-j` independent ordering (Ninja output is content-deterministic; check generated sources sorted), `strip` and no embedded git describe with dirty state or timestamps. LTO: GCC `-flto=N` with `-frandom-seed` is usually reproducible but is a known flake source [K]. Python/other generators in the build: sort dict outputs.

HIP/ROCm: no source I found states that hipcc/clang device output is reproducible or not [S, negative result for several searches]. What I found:
- HIP fat binaries are made by `clang-offload-bundler`; code objects may be compressed ("CCOB", zstd) in `.hip_fatbin` since ROCm 6.x [S ROCm/legacy-rocm-build issue 4932]. Alignment requirement `-bundle-align=4096` [S PyTorch rocm_multiarch_utils].
- openSUSE had to patch ROCm `clr` so that `hip_pch.o` (the HIP precompiled header) contains no randomness (ROCm/clr PR #97) [S src.opensuse.org rocclr pulls/1, 2025-10]. That is a runtime-library build artifact, not our device code, but it shows non-determinism exists somewhere in the ROCm build.
- Spack docs advise pinning exact ROCm versions in paths for reproducible results [S]; not about bytes.
**[E] own experiment, 2026-10-28, this machine**, toolchain `work/rocm-10.1`, `AMD clang version 24.0.0git (ROCm/llvm-project d6f6cb69)`: a trivial HIP kernel (`k.hip`, one `__global__` function, `-O2 -x hip --offload-arch=gfx1200`, `-ffile-prefix-map=<srcdir>=.`) compiled twice in different directories, 2 s apart, then linked (`-Wl,--build-id=sha1`, RUNPATH to the venv): `k.o` and the linked executable are **bit-identical** (sha256 `7c50fa8f...` and `bf23cbbb...`). Without `-ffile-prefix-map` the object differs when the source path differs (`9121fe37...` vs `96e1a2c0...`), as expected. Scratch files: `a scratch folder (repro/)`. Limits: one 3-line kernel, no `-flto`, no zstd compression flag, no real engine, nothing run on the GPU. It shows the AMD clang 24 code path has no timestamp/randomness for that case. It does **not** prove the 100+ kernel engine is reproducible (open item 1).
Tools [K, not fetched]: `diffoscope` (compares ELF sections, can unpack `.hip_fatbin` only as raw bytes; use `clang-offload-bundler --list/--unbundle` and `llvm-objdump`/`readelf` on extracted code objects to localize a diff), `reprotest` (varies path, time, locale, user, umask, CPU count, build ordering).
Practical recipe: build in a fixed container image identified by digest, fixed path `/build`, `SOURCE_DATE_EPOCH=<commit time>`, `LC_ALL=C`, `TZ=UTC`, `-ffile-prefix-map=/build=.`, `-Wl,--build-id=sha1`, `-frandom-seed`; run twice in CI, compare; run once locally by the maintainer, compare to CI artifact. If the fatbin proves non-deterministic, option: ship device code objects as separate files (`--offload-device-only`) whose hashes are compared individually, or exclude compression (`-fno-gpu-compress` [?] flag name unverified) and see what remains.

### 4. Signature verification inside C++

**minisign format** [V https://jedisct1.github.io/minisign/, fetched 2026-10-28]:
```
untrusted comment: <text>
base64( algo(2) || key_id(8) || signature(64) )
trusted_comment: <text>
base64( global_signature(64) )
```
algo `Ed` = Ed25519 over the file; `ED` = Ed25519 over **BLAKE2b-512(file)** (prehashed, default, "new implementations must use the hashed format"); global signature = `ed25519( <signature(64)> || <trusted_comment bytes> )`. Public key line: base64(`Ed` || key_id(8) || pk(32)). Page gives no numbered verification procedure; steps below are my inference from the format and the reference implementation [K], mark for test vectors:
1. Parse 4 lines; reject if line 2 is not exactly 74 bytes decoded (2+8+64); algo must be `ED` (reject `Ed`).
2. key_id must equal the key id of an embedded key (select the key by id; do not try all).
3. h = BLAKE2b-512(file), streaming (libsodium `crypto_generichash` with outlen 64, no key).
4. Verify signature over h with that key (plain Ed25519, not Ed25519ph).
5. Trusted comment: strip the `trusted_comment: ` prefix (bytes after the prefix to end of line, no newline); verify global_signature over `signature || trusted_comment`.
6. Only then trust the trusted comment (put `seq`, `expires`, `product`, manifest hash there if desired) and the file.
Pitfalls: untrusted comment is not authenticated; verifying only the file signature and skipping the global one leaves the trusted comment (and thus version/timestamp) forgeable; legacy `Ed` must be refused (also avoids loading whole file); compare key ids; line endings; reject extra lines/trailing data; cap sizes; verify before JSON parse.

**Libraries**
- libsodium: `crypto_sign_verify_detached`, `crypto_generichash` (BLAKE2b). Docs page [V https://libsodium.gitbook.io/doc/public-key_cryptography/public-key_signatures]: returns 0/-1, deterministic Ed25519, `ed25519ph` variant exists; the page says nothing about malleability/small-order checks [V]. From [K]: libsodium's detached verify rejects non-canonical S and small-order R/pk; keep that claim marked unverified and test with RFC 8032 / Wycheproof vectors. Large (about 300 KB), widely packaged on Fedora (`libsodium-devel`), also what the minisign reference tool uses [V earlier report].
- Monocypher: single C file pair; Cure53 audit of v3.1.1 completed June 2020 (focus: core + optional Ed25519 header) [S opentech.fund, cure53 pentest-report_monocypher.pdf; full PDF not read]. The brief's "2017/2023" audits: I found only the 2020 one [?]. Ships BLAKE2b (`crypto_blake2b`) and EdDSA with BLAKE2b by default; **the default Monocypher EdDSA uses BLAKE2b instead of SHA-512, so it is NOT Ed25519-compatible; the `monocypher-ed25519.h` optional file provides RFC 8032 Ed25519 with SHA-512** [K]. Easy to vendor (two files, public domain/CC0 + BSD) = best "small auditable".
- OpenSSL 3: Ed25519 is PureEdDSA via `EVP_DigestVerifyInit` with NULL digest and one-shot `EVP_DigestVerify`; context strings not allowed for Ed25519 [V https://docs.openssl.org/3.5/man3/EVP_DigestVerifyInit; search summary]. BLAKE2b-512 is available as `EVP_blake2b512()` [K, not verified this run]. Already present if libcurl uses it, but huge API surface and provider/config loading; no advantage for a 300-line verifier.
**Recommendation:** vendor Monocypher (+ `monocypher-ed25519.h`) for Ed25519 and SHA-512/256, or link libsodium if it is already a dependency. Either way write ~150 lines of format parsing, test against minisign/OpenSSH vectors, fuzz the parser (libFuzzer). Constant-time is not required for verification (all inputs public) [K].

### 5. Key management and signing scheme comparison

| | minisign | OpenSSH `ssh-keygen -Y sign` (SSHSIG), sk keys | Sigstore keyless |
|---|---|---|---|
| Private key location | file, passphrase-encrypted (scrypt) [V minisign README via earlier report]; **no hardware token support found** (search found nothing, README not checked for PKCS#11) [S/?] | `ed25519-sk` on FIDO2 YubiKey (firmware >= 5.2.3 for Ed25519) [S developers.yubico.com]; `touch-required`, `verify-required` (PIN), `resident` options [V https://man.openbsd.org/ssh-keygen.1]; key cannot be copied | no long-lived key; short-lived cert bound to OIDC identity (GitHub/Google account) |
| Verify tool | `minisign -V`, tiny format | `ssh-keygen -Y verify -f allowed_signers -I principal -n namespace -s sig` [V man page]: namespaces, `valid-after/valid-before`, revocation file `-r`, `cert-authority`; sk types accepted in allowed_signers | cosign / gh; **no C++ verifier found** [V earlier report]; needs TUF-delivered trusted root, Rekor proof |
| Threshold | not native | not native (multiple `-Y sign` calls, client counts) | not native |
| Theft of GitHub account | irrelevant to signature | irrelevant | **fatal if identity = GitHub account**: "if an identity provider or account is compromised, Sigstore still issues certificates to it"; OIDC compromise is out of Sigstore's scope [S docs.sigstore.dev/threat-model, blog playbook]. Detection only via transparency log monitoring |
| C++ cost | ~150 lines | ~250 lines SSHSIG framing + sk signed-blob layout [K: SHA256(app) \|\| flags \|\| counter \|\| SHA256(msg)] or shell out to `ssh-keygen` | unrealistic |

Notes:
- The sk (hardware) signature layout in OpenSSH is from PROTOCOL.u2f / PROTOCOL.sshsig [K, not fetched; verify against the OpenSSH source before implementing]. Verification does not need the token; only signing does.
- Yubico documents git/SSH signing with FIDO2 keys [S developers.yubico.com/SSH/Securing_git_with_SSH_and_FIDO2.html]. I found no OSS project signing release artifacts this way [S negative] => novel, so test vectors from our own `ssh-keygen` are the safety net.
- Simplest safe option if complexity worries you: **minisign key on an air-gapped/offline machine** (the key never touches a networked host). Weaker than hardware-resident, because file theft + passphrase guess is possible, but no framing code.
- age is encryption, not signature; skip. GPG: heavy, rejected in earlier report; CPython moves away from PGP (PEP 761) [S peps.python.org/pep-0761]; Fedora downstream complained about cosign availability [S discuss.python.org], a hint that a C++ client should not depend on cosign either.

**Recommended scheme:** SSHSIG with `ed25519-sk` (touch + PIN) keys, verified natively; namespace string `tona-release-v1`; fall back to minisign-offline if the SSHSIG verifier slips. (Alternative: sign with both, and accept either. Not recommended: more code, same trust.)

### 6. Client-side update hardening

1. **Never root.** All under `~/.local/share/tona/`, files mode 0700/0600, unit stays a `--user` unit.
2. Private download dir `~/.local/share/tona/staging/<random>/` created with `O_EXCL`, `0700`, on the same filesystem as `versions/` (atomic `rename`). Not `/tmp` (tmpfs on this machine; also symlink races).
3. libcurl: HTTPS only (`CURLOPT_PROTOCOLS_STR "https"`, `CURLOPT_REDIR_PROTOCOLS_STR "https"`), `CURLOPT_SSL_VERIFYPEER/HOST` defaults on, `CURLOPT_MAXFILESIZE_LARGE` from the signed size, `CURLOPT_LOW_SPEED_LIMIT/TIME`, `CURLOPT_TIMEOUT`, max redirects 3, **no cert pinning needed**: trust is the signature; TLS is only privacy/availability. [K for option names; confirm in curl docs]
4. Order: fetch manifest -> size-cap -> verify signature(s) -> parse JSON (strict, no duplicate keys) -> check `product`, `seq > stored`, `expires`, `min_installer` -> download artifact -> hash while streaming -> compare SHA-256 and size -> only then open the archive.
5. Safe extraction (tar.zst): do not shell out to `tar` on untrusted input without flags; write own extraction or use libarchive with `ARCHIVE_EXTRACT_SECURE_NODOTDOT | _SECURE_SYMLINKS | _SECURE_NOABSOLUTEPATHS`, reject symlinks/hardlinks/devices/setuid, all entries under one top dir, cap total unpacked size and entry count, reject names with `..`, absolute paths, NUL, duplicates; extract into a fresh staging dir; `fchmod` normalized modes; use `openat` relative to dirfd with `O_NOFOLLOW`. Because signature precedes extraction, the archive is trusted-author content; this is defence in depth against a signing mistake. [K; libarchive flag names to verify]
6. Build-from-source fallback sandbox: `systemd-run --user -p PrivateNetwork=yes ...`. Findings: for user managers `PrivateUsers=` is implicitly enabled and requires unprivileged user namespaces [V systemd.exec man page fetched 2026-10-28 (first 100k chars)]; if user/network namespaces are disabled (e.g. `user.max_user_namespaces=0`) PrivateNetwork fails [S apptainer docs]. I could not confirm from a primary source that `PrivateNetwork=yes` works in `--user` mode; a failure report exists for Debian 13 with "Failed to set up user namespacing" [S]. **Test on this Fedora 44 box before relying on it** (open item 4). Alternative: `bwrap --unshare-net --ro-bind ... --bind build ...` [K] (bubblewrap is a Fedora dependency of Flatpak, widely installed). Build step gets: read-only source and toolchain, writable build dir, no network, no `~/.ssh`, no `~/.config/tona/api-key`.
7. Atomic switch: new `versions/<ver>-<hash8>/` complete + `fsync`, then `symlink(tmp)` + `rename(tmp, current)` [K]. Keep N=3 versions. Persist `last_accepted_seq` and the manifest file only after switch succeeded.
8. Health check: restart, poll `/status`, timeout; on failure swap symlink back (rollback). Never run blocking HIP calls in the updater (project memory: GPU hang kills desktop).
9. Alternative channel: signed git tags. `git verify-tag` with `gpg.format=ssh` and `gpg.ssh.allowedSignersFile` works with sk keys [K; Yubico guide S]. Good for the source-build path; same root key list. Note the git-ssh namespace is `git`, use a different namespace than the manifest so a signed commit cannot be replayed as a manifest [K].
10. Update check privacy and no auto-install without consent (earlier report).

### 7. GitHub account and repo hardening

- 2FA with hardware keys (WebAuthn), at least two registered, recovery codes offline. Disable SMS fallback. [K; docs not fetched]
- Rulesets: protect `v*` tags (restrict creation/update/deletion; block force pushes) [S Eclipse blog "Don't become the next Trivy"]; branch rule on `main`: PR required, signed commits, no force push, linear history.
- Immutable releases on (above). Draft -> attach assets -> publish.
- Environment `release` with required reviewer and deployment limited to tags `v*`; secrets only after approval; "prevent self-review" [V https://docs.github.com/en/actions/managing-workflow-runs-and-deployments/managing-deployments/managing-environments-for-deployment]: up to 6 reviewers, free plan allows environments on public repos only [V]. With one maintainer a required-reviewer rule is only a speed bump (same account approves); it becomes real with a second maintainer. Therefore **do not put the release key in any environment at all.**
- Workflow permissions default `contents: read`; `id-token: write, attestations: write` only on the attest job; actions pinned by SHA (+ Dependabot/pinact to update) [S classmethod pinact, wiz]; no `pull_request_target`; build job has no secrets; fork PRs get nothing.
- OIDC: used only for attestations, not for cloud access here.
- **If the GitHub account is stolen:** A: attacker pushes poisoned source/tag or publishes a new release; clients reject: no valid signature (manifest key offline). Source-build users who `git pull` without `git verify-tag` are exposed, so the documented build path must verify tags. B: attacker controls CI => can sign if key is in CI => total loss; if only attested, attestation says "valid workflow" => clients using only attestations accept the malware. C: attacker can publish unsigned artifacts (rejected), can delete releases (DoS), cannot make existing signed manifests invalid; recovery = new account/repo URL embedded in a signed manifest, or ship the new repo URL via a release signed by the root key (keep `repo_url` in the manifest, but do not auto-follow changes without user confirmation).

### 8. Signing key location, who may use it, and more maintainers later

**Day-one client design (do this now so no deployed client breaks later):** the client embeds `root.json` v1 = `{version, expires, keys: [{id, type, pk}], threshold, revoked: []}` and verifies *every* manifest as "at least `threshold` distinct valid signatures among `keys`". Today: 2 keys (primary YubiKey, backup YubiKey in a safe), threshold 1. Later: add maintainers, raise threshold, with no client change.
- Rules (TUF-style): count distinct key ids only (go-tuf had a bug where one key under several ids counted multiple times towards the threshold [S fuchsia go-tuf v0.3.2 backport]); signatures per manifest are an array; unknown keys ignored.
- Root rotation (add/remove maintainer, change threshold, replace key): new `root.json` with `version = old+1`, signed by a threshold of the **old** keys *and* a threshold of the **new** keys, delivered as a normal release artifact next to the manifest; client walks root v(n)->v(n+1)->...; every step needs its own signatures; client stores the highest accepted root version. This is TUF's root update flow [V TUF spec: root version increments by exactly one; Foundries doc: new root.json signed by the previous key S]. No trust in GitHub needed: only the signatures matter. Compromised-key revocation: rotate root without that key; clients that are offline for long must catch up through the whole chain (keep all roots in each release, a few KB).
- Day-one limits: a lone maintainer losing both tokens = project root lost; users must reinstall from a new out-of-band root. Mitigate with a third key (paper-backed or an offline minisign/SSH key in a bank deposit), thresholds 1 of 3 until a second human exists, then 2 of 3 (second maintainer's token + one of yours).
- Who may use the key: only the maintainer, physically, via touch+PIN. CI never holds it. A second maintainer later gets their own token; adding them is a root rotation (threshold unchanged until they join, then raised).
- Separation of duties: CI = build + attest + upload to draft. Maintainer = independent rebuild + review + sign + publish. Release publishing is a human action after the signature exists.
- Multiple independent builders (Bitcoin Core model): each builder runs the deterministic Guix build, signs a hash manifest and commits it to `guix.sigs`; consumers run `guix-verify`/`contrib/verify-binaries` which needs matching hashes [S github.com/bitcoin-core/guix.sigs, bitcoin/bitcoin contrib/guix/README.md, PR #21462, #22182; v22.0 (2021-09) was first fully Guix release]. For us: maintainer rebuild now; later any contributor can add a signed `builder-attestations/<ver>/<name>.sig`. A threshold of "k builders agree on the hash" can later be enforced in the client (k-of-n builder keys) [design idea, not an existing standard].
- Others: Tor Browser: several developers' keys plus a group key, signing keys backed up/deployed on dedicated signing machines (tor-browser-build commit 2026-07/08, single source) [S]; deterministic builds history [S]. CPython: individual release managers sign; PEP 761 moves to Sigstore (keyless, identities of release managers) [S peps.python.org/pep-0761, sethmlarson.dev]. Go: sum.golang.org transparent log (Merkle tree, Trillian), clients check inclusion and consistency proofs and so can use an untrusted proxy [S go.dev/blog/module-mirror-launch]. Sigsum (minimal transparency log with witnesses) as a lighter alternative [S git.sigsum.org design.md; not evaluated]. Debian/Arch keyrings, Homebrew, rustup/crates (TUF work), Tails: **not researched** [?].
- Sigstore keyless vs long-lived key for a maintainer team: keyless removes key handling but makes the OIDC account the root of trust; stolen GitHub account => valid signatures for the attacker, and the only detection is monitoring the Rekor log for the maintainer identity (rekor-monitor scans index ranges and reports entries matching given identities or key fingerprints; Trail of Bits/OpenSSF preparing it for production; runnable hourly as a GitHub Action) [S blog.sigstore.dev/using-rekor-monitor, blog.trailofbits.com 2025-12-12]. Monitoring runs on GitHub itself, which the attacker may control => run it elsewhere. For this project keyless is **not** the root of trust; it may be added later as an *extra* attestation/witness.
- Hardware: YubiKey FIDO2 `ed25519-sk` (needs firmware >= 5.2.3 [S Yubico]); OpenPGP card/PIV alternatives are possible but add gpg/pkcs11 for signing; for verification they still produce Ed25519 signatures if the card supports it [K, unverified]. Two tokens enrolled (primary daily, backup stored elsewhere); non-resident handles must be backed up too (the key handle file is needed with non-resident sk keys; losing it loses the key even with the token [K, verify]). Prefer `resident` only with a PIN and accept the theft-risk trade-off the man page notes [V].
- Ceremony (single maintainer): generate sk keys on the offline machine, record public keys in `root.json` v1, sign root v1 with all keys, print fingerprints on paper, publish fingerprints in README, website, a second GitHub account/gist and a signed Git tag, so a user can cross-check out-of-band. Test restore of the backup token before first release. Passphrase/PIN not stored on the networked machine.

### 9. Detection

- Append-only public record of each release: tag signed + manifest in release + `transparency/<ver>.json` committed (signed commit). Cheap alternative to Rekor: users' TUI can compare `seq` and manifest hash against a second host (e.g. a static site) [design idea].
- A second person/machine running `verify-release` in cron is the only defence against targeted redirection like Notepad++ (a per-user attack is invisible to all but the victim). Client mitigation: manifest fetch from two origins (GitHub + project domain), warn if they differ [design idea].

### 10. Recommended design (C)

**Signed objects**
- `manifest.json` (<= 64 KiB): `{product, version, seq, issued, expires (<= 30 d), repo_url, root_version, channel, severity, artifacts: [{name, size, sha256}], source_tarball sha256, git_commit, toolchain_lock_sha256, models_toml_sha256, min_installer, changelog}`.
- `manifest.json.sig` containing an array of SSHSIG blobs (namespace `tona-release-v1`) or, fallback, minisign `ED` signatures with the trusted comment duplicating `seq`+`version`+SHA-256 of manifest; both verified natively.
- `root.json` (+ `root.json.sig`) as in section 8; all roots since v1 shipped with each release.
- Source tarball = `git archive --format=tar --prefix=tona-<ver>/ <tag>` piped through deterministic `zstd` (fixed level, `-T1`), reproduced by CI and maintainer.

**Release workflow (in order)**
1. Maintainer: `git tag -s` (SSH sk-signed tag) on a commit of `main` that passed `check.yml`, BENCHMARKS gate done. Tag ruleset allows creation only by the maintainer.
2. Tag push triggers `release.yml` (all actions SHA-pinned, `contents: read`, no secrets, no PR triggers): checkout tag, verify `git verify-tag` against allowed_signers committed in a protected path, `git archive` tarball, build in the digest-pinned Fedora container with the hash-checked ROCm lock, run tests that need no GPU, build twice and compare, produce `SHA256SUMS` and artifacts, `actions/attest` (needs `id-token`, `attestations: write`) in a separate job, upload to a **draft** release. CI never has the signing key.
3. Maintainer (offline-ish machine, ideally a clean checkout): download draft assets, `gh attestation verify` (provenance only), **rebuild locally in the same container, compare SHA-256 of every artifact; mismatch => stop, run diffoscope, no release**. Also verify `git archive` hash == tarball.
4. Maintainer builds `manifest.json` (increment `seq`), signs with the sk key(s) (touch + PIN), runs the real client verifier against the draft files (dry run).
5. Upload manifest + sig + roots to the draft, publish (immutable). Enable "latest" only now.
6. Post-publish: a second machine/CI-from-another-account runs `verify-release <tag>` (re-downloads from the public URL, checks signatures); update fingerprint page if keys changed.

**Client steps (in order)**
1. Opt-in check; GET `releases/latest/download/manifest.json` and `.sig` [rate-limit behaviour [K, unverified]]. Size cap, timeouts, HTTPS only.
2. Verify root chain from embedded `root.json` v1 to the newest offered root; then verify manifest with `threshold` distinct keys; strict parse.
3. Check `product`, `seq > last_accepted_seq`, `expires > now` (stale => warn, do not install), `min_installer`, `root_version`.
4. Show changelog/size/severity; user confirms.
5. Download artifacts to staging with Range resume, streaming SHA-256, exact size; reject mismatch; delete staging.
6. Safe extraction into `versions/<ver>-<hash8>/` (section 6.5). No scripts from the archive are executed.
7. Smoke test (no GPU hang risk: use only what already exists, run the binary with `--version` and a CPU self-test) -> atomic `current` swap -> restart user unit -> health poll -> commit `last_accepted_seq`; on failure rollback automatically.
8. If the user has no matching ROCm lock: stage toolchain per `toolchain_lock_sha256` (pip `--require-hashes`) before the swap. If the user opts for source: `git verify-tag` with the embedded allowed_signers, build in the sandbox (section 6.6), compare the resulting binary hash with the manifest and warn if different (shows reproducibility drift, not an attack by itself).

**Client binary is itself the bootstrap risk:** first install via `curl | sh` or a `git clone` can be attacked at that moment. Provide: fingerprint in README + signed git tag verification instructions + `tona-install` tarball verifiable with `ssh-keygen -Y verify` by hand [design].

### Open / verify first (gating items)
1. Is the full engine build bit-reproducible (host GCC + hipcc, all kernels, with and without `-flto`, with code object compression)? Run: build twice at different paths/times, `diffoscope`. My [E] test covers one kernel only.
2. SSHSIG + sk signature layout (PROTOCOL.sshsig / PROTOCOL.u2f): read the OpenSSH source and produce test vectors before writing the verifier.
3. libsodium/Monocypher strictness on non-canonical S, small-order points; Monocypher audit coverage of the Ed25519 header (2020 audit mentions it [S]).
4. `systemd-run --user -p PrivateNetwork=yes` and bubblewrap `--unshare-net` on this Fedora 44 box.
5. `releases/latest/download/` and API rate limits; GitHub 2FA/ruleset doc pages not fetched this run.
6. Not researched: Debian/Arch keyrings and WKD, Homebrew, rustup/crates TUF work, Tails, Sigsum maturity.
7. Unconfirmed claim in the brief: "cache poisoning" in Ultralytics (sources say script injection).

### Sources used (all fetched/searched 2026-10-28)
TUF spec https://theupdateframework.github.io/specification/latest/ [V]; minisign https://jedisct1.github.io/minisign/ [V]; ssh-keygen https://man.openbsd.org/ssh-keygen.1 [V]; immutable releases https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases [V]; GH environments https://docs.github.com/en/actions/managing-workflow-runs-and-deployments/managing-deployments/managing-environments-for-deployment [V]; reproducible-builds https://reproducible-builds.org/docs/build-path/ and /source-date-epoch/ [V]; libsodium https://libsodium.gitbook.io/doc/public-key_cryptography/public-key_signatures [V]; systemd.exec https://www.freedesktop.org/software/systemd/man/latest/systemd.exec.html [V partial]; OpenSSL https://docs.openssl.org/3.5/man3/EVP_DigestVerifyInit [S]; xz https://openwall.com/lists/oss-security/2024/03/29/8 and https://arxiv.org/html/2404.08987v1 [S]; tj-actions https://threats.wiz.io/all-incidents/tj-actionschanged-files-supply-chain-attack [S]; Trivy https://wiz.io/blog/trivy-compromised-teampcp-supply-chain-attack [S]; Ultralytics https://www.hiddenlayer.com/research/ultralytics-python-package-compromise-deploys-cryptominer [S]; Notepad++ https://www.theregister.com/2026/02/02/notepad_plusplus_intrusion/ , https://thehackernews.com/2026/02/notepad-fixes-hijacked-update-mechanism.html [S]; Monocypher audit https://www.opentech.fund/security-safety-audits/monocypher/ [S]; Bitcoin guix.sigs https://github.com/bitcoin-core/guix.sigs [S]; rekor-monitor https://blog.sigstore.dev/using-rekor-monitor/ , https://blog.trailofbits.com/2025/12/12/catching-malicious-package-releases-using-a-transparency-log/ [S]; Sigstore threat model https://docs.sigstore.dev/threat-model/ and playbook https://blog.sigstore.dev/dont-panic-a-playbook-for-handling-account-compromise-with-sigstore-ee299dca5144 [S]; PEP 761 https://peps.python.org/pep-0761/ [S]; Go sumdb https://go.dev/blog/module-mirror-launch [S]; Yubico https://developers.yubico.com/SSH/Securing_git_with_SSH_and_FIDO2.html [S]; Eclipse tags/rulesets https://blogs.eclipse.org/post/mika%C3%ABl-barbero/dont-become-next-trivy-how-make-your-releases-tags-and-automation-resistant [S]; go-tuf threshold fix https://fuchsia.googlesource.com/third_party/go-tuf/+/refs/tags/v0.3.2%5E%21/ [S]; Foundries TUF offline keys https://docs.foundries.io/96/reference-manual/security/offline-keys.html [S]; ROCm CCOB issue https://github.com/ROCm/legacy-rocm-build/issues/4932 [S]; openSUSE rocclr reproducible patch https://src.opensuse.org/ROCmWork/rocclr/pulls/1.patch [S].

## Decision (2026-10-28)

The design above (hardware-backed keys, k-of-n threshold, mandatory rebuild before signing) is more than a project
of this size needs at the start. Chosen instead:

- GitHub Releases as the only channel, immutable releases, artifact attestations from CI.
- The maintainer signs a manifest (version number, expiry, SHA-256 and size per file) with minisign, one command per
  release. The key is a password-protected file kept by the maintainer with a backup copy, never a CI secret. Reason:
  a CI signature only proves "this workflow built it" and does not protect against a taken-over account, a leaked
  token or a compromised action; the maintainer signature does.
- `tona-top` embeds two public keys (main and backup) so a key can be replaced later without breaking installs, and
  verifies the signature before it parses or extracts anything; it never installs a lower version number.
- Rebuilding and comparing hashes before signing is an optional script; the builds are already bit-identical.
- The maintainer's GitHub login uses a passkey or hardware key (phishing-resistant), not a TOTP app.
