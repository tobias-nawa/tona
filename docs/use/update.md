# Updates

`tona` checks for a release, installs it, restarts the server and goes back if it does not start. Release installation
only (`$XDG_DATA_HOME/tona`, default `~/.local/share/tona`), and only if you switch it on. A checkout updates with `git pull`
and `just build`.

## Turning it on

The setup asks once, on its Finish screen, with the box "Check for updates once a day" ticked (it only checks and tells
you; nothing installs without your ok). `--updates on|off` does the same without a screen. `c` twice on the Start page of `tona` changes it (stored in `~/.local/share/tona/state.json`). With checks on,
`tona` looks at start and then at most every 24 hours: it downloads the manifest and its signature from the GitHub
release page and sends nothing else (GitHub sees your IP). A newer release shows a hint; `u` opens a dialog with size
and changelog and asks before it installs.

## Without the screen

```sh
tona update              # check, show version and size, ask, install, restart
tona update --yes        # the same without the question
tona update --rollback   # back to the previous version, asks first
```

## What `u` does

1. Re-fetches and checks manifest and signature; a release that fails is dropped.
2. Downloads every file into a private staging directory; size must not exceed the manifest's, SHA-256 is checked on
   arrival and after reading back.
3. Moves the version into `versions/<version>/`, points `previous` at the running version and `current` at the new one
   (atomic renames), then records the sequence number in `state.json`. Versions beyond the newest three are deleted, never
   `current` and `previous`.
4. Waits until no request runs, restarts the server and polls `/status` until ready (180 s at most). Linux:
   `systemctl --user restart tona-server`. macOS: ends the `tona-server` of `server.pid` and starts `tona start --foreground`;
   if another server answers on the port, it starts nothing and says so.
5. If the new server exits or does not become ready, points `current` back at `previous`, restarts and tells you. A
   server that exited is noticed within a second. The release that failed is not offered again.

If requests keep running, the new version is installed but not started, and `tona` says so.

## What is checked

- A minisign signature (prehashed, BLAKE2b-512) of one of the two public keys compiled into `tona`; without it nothing is
  read. The signed comment must be `tona schema=1 seq=<seq> version=<version>`.
- Strict manifest: unknown or repeated keys, wrong types, paths other than `bin/<name>` or `runtime/<name>`, modes other
  than 0755 and 0644, more than 256 files or 4 GiB are errors.
- Every file names its platform (`linux-x86_64-gfx1200`, `macos-arm64`, `any`); a client installs its own and `any`, and
  needs a `bin/tona-server`. The Mac release has no `runtime/`.
- The manifest must not be expired, and its sequence number must not be lower than the highest accepted (an older signed
  manifest is refused).
- HTTPS only, certificate and host checked, at most three redirects, only to https.
- The installer refuses root and install directories others can write to, never follows a symlink while writing, and
  unpacks no archive.

The signing key stays with the maintainer; CI cannot sign.

## When something fails

The running version stays; the message shows in the Update box and footer, nothing under `versions/` or `current` changes,
and the check retries at the next start or after 24 hours.

| Message | Meaning |
|---|---|
| `signature check failed: ...` | not signed by a trusted key, or altered; do not install |
| `manifest rejected: ... older than the last accepted` | an old release is offered; ignore it |
| `manifest rejected: ... expired` | check your clock or wait for a new release |
| `install failed, nothing changed: ...` | a download did not match size or hash, or the disk refused |
| `new version failed (...); the previous one runs again` | the new server exited or did not become ready; the old one was restored |
| `install root is open to others; run:` and `chmod 700 <dir>` on the next line | a folder you made yourself has the wrong mode |
| `installed runtime does not match this release` | delete `runtime/rocm-<id>` and update again |

## Going back by hand

`R` twice on the Start page, `tona update --rollback`, or:

```sh
cd ~/.local/share/tona
ln -sfn "$(readlink previous)" current.new && mv -T current.new current
systemctl --user restart tona-server
```

## Checking a release by hand

Verify `manifest.json` and `manifest.json.minisig` from the release page with the key starting with `RW` in
`engine/update/keys.h`:

```sh
minisign -V -H -P <public key> -m manifest.json
```

`-H` refuses the old non-prehashed format. The `trusted comment:` must read `tona schema=1 seq=<n> version=<version>`;
`sha256sum` of each asset must equal its `sha256` entry.

## Layout

`tona setup` makes the first installation and writes the unit ([release.md](../maintain/release.md#installation-without-a-checkout));
an update never writes it.

```
~/.local/share/tona/                 mode 0700
  versions/<version>/bin/          the programs of one release
  versions/<version>/lib           link to ../../runtime/rocm-<id>
  versions/<version>/manifest.json, manifest.json.minisig   as signed
  runtime/rocm-<id>/               ROCm libraries, shared by versions with the same toolchain lock
  current, previous                links to a version; the unit runs current/bin/tona start --foreground
  models/, profiles/, prompt-cache/  written by tona setup; not touched by updates
  staging/                         emptied at the start and end of every update
  state.json                       highest accepted sequence number, whether checks are on
```

Variables: [environment.md](environment.md#updater). Tests: `ctest -L unit` runs `update_*` against a server on
127.0.0.1; `python3 engine/tests/data/update/generate.py --minisign` checks the vectors against real `minisign`;
`scripts/fuzz.sh` fuzzes the parsers.
