# Changelog

All notable changes to Tona are written here, newest first, in the form of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The text of a version is what the release page shows and
what `tona` shows in its update dialog, so write for the user, in short lines. To release, run
`scripts/release-next.py` (it computes the version and drafts the section from the pull request titles), edit the
section for users and keep an empty `[Unreleased]` above it; the release script takes the text of the
version from here and fails without it (docs/maintain/release.md).

## Versioning

Tona follows [Semantic Versioning](https://semver.org/). While the major version is 0, the public interface is not
frozen: **a minor version (0.Y.0) may contain breaking changes**, and each one is listed under "Changed" or "Removed"
with what to do. A patch version (0.Y.Z) fixes bugs and does not break the interface.

The version is computed from the titles of the pull requests, which are Conventional Commits
(`scripts/release-next.py`, `tools/gates/pr_title.py`): while the major version is 0, a breaking change (`!` in the
title or a `BREAKING CHANGE:` paragraph) or a `feat` raises the minor version, a `fix` or `perf` the patch version.
From 1.0.0 a breaking change raises the major version, a `feat` the minor, a `fix` or `perf` the patch. `docs`, `test`,
`refactor`, `build`, `ci` and `chore` alone make no release. The first release is 0.1.0.

What counts as the public interface:

- the commands and options of `tona` and `tona-server`,
- the keys of the settings file `~/.config/tona/tona.conf` and the environment variables documented in `docs/`,
- the HTTP API (OpenAI- and Anthropic-compatible endpoints) of `tona-server`.

Not part of it: output texts of the programs, the layout of the user interface, internal file formats, the files
under `research/`, the code of the engine, and measured speeds.

## [Unreleased]

### Added

- `tona` asks on its first start (and again with `tona setup`): it detects your hardware, asks what you want to do and in which programming languages, recommends one
  model per task with its measured speed, strengths and limits, downloads it and sets up the server as a systemd
  user unit. Autostart is off unless you choose it.
- `tona`, the control centre: running conversations with decode and prefill rate, memory, last requests, server
  start and stop, and a launcher for coding agents that talk to the local server.
- `tona` is the one program you download: setup, control centre, commands without a screen, the coding agents, the
  background server, `tona update` and `tona uninstall`; it needs neither bash nor `jq`.
- `tona-server`: OpenAI- and Anthropic-compatible API on `127.0.0.1:8662`, optional API key.
- Three models: Coder (Qwen3.8-Flash-Next, IQ1_M) for coding agents, Ornith 1.5 35B-A3B (Q4_K_M) for coding, strong
  in Rust, and Nemotron 3.5 Bolt 74B-A9B (2-bit) for knowledge and research. Numbers and limits: README and
  `BENCHMARKS.md`.
- Updates you switch on yourself: `tona` checks for a new release (two small files, no identifier) and installs
  it only after you confirm. Every release is signed with minisign and carries build attestations.

- Experimental: macOS 15 or newer on Apple silicon (Metal). The release carries prebuilt `tona-server`,
  `tona` and the Metal kernels as `macos-arm64-*` files; `tona` installs them into `~/.local/share/tona`
  (`tona start`, `tona stop`, `tona update`, optional autostart at login). Not part of the Mac release yet:
  notarization (a browser download needs one command, see `docs/use/macos.md`).

### Security

- Releases are installable only with a valid minisign signature of `manifest.json` (published as
  `manifest.json.minisig`); the installer and `tona` check it against keys compiled into the program, then size
  and SHA-256 of every file. How to check by hand: `docs/maintain/release.md`.
- Model files are checked against pinned SHA-256 values; a wrong file never gets its final name.

### Known limits

- Supported: AMD Radeon RX 9060 XT (16 GB) with 64 GB of DDR5 on Linux. Apple silicon with macOS 15 or newer is
  experimental (see above). Other hardware is not supported yet.
- Not built yet, no date promised: reading documents and images, audio to text, reading aloud, image generation.
