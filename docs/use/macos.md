# Building on macOS

The shared part builds on Apple silicon and its unit tests pass. The Metal backend is experimental (tier 3, off by
default): `engine/backend/metal/AGENTS.md`.

## Install from a release

Prebuilt programs for macOS 15 or newer on Apple silicon; no Homebrew or Xcode needed. They are signed ad hoc, not
notarized.

```sh
curl -fL -o tona https://github.com/tobias-nawa/tona/releases/latest/download/macos-arm64-tona
chmod +x tona
./tona
```

A file downloaded with a browser is quarantined ("cannot be verified"); after checking it
([release.md](../maintain/release.md#check-a-release-by-hand-users)) remove the flag once, or download again with `curl`:

```sh
xattr -d com.apple.quarantine tona
```

The setup names the Mac's class by memory (32 GB, 64 GB and more) and offers the Coder; [classes](installer.md#classes).

The setup verifies the maintainer's signature and every SHA-256 first, then:

- installs `tona-server`, `tona.metallib`, `libomp.dylib`, `tona` and the licence texts into
  `~/.local/share/tona/versions/<version>/bin/` and points `current` at it;
- downloads model and expert profile, writes the API key and `~/.config/tona/tona.conf`;
- links `tona` into `~/.local/bin` and offers to add that folder to `PATH` ([Finish](installer.md#finish)).

`tona start` runs the server in the background (log `~/.local/state/tona/server.log`), `tona stop` stops it. Start at
login is off unless you choose it in the setup or run `tona autostart on` (a LaunchAgent,
`~/Library/LaunchAgents/com.tobiasnawa.tona.plist`). `tona` updates as on Linux ([update.md](update.md)). `tona uninstall`
removes Tona.

## Build from source

Requirements:

- macOS 15 or newer on Apple silicon (Metal Shading Language 3.2); Xcode or Command Line Tools, CMake 3.28+, Ninja, just
- `brew install libomp` (`engine/macos_openmp.cmake` looks under `$HOMEBREW_PREFIX/opt/libomp`)
- `brew install openssl@3` and `-DOPENSSL_ROOT_DIR=$(brew --prefix openssl@3)` (`openssl@4` breaks `httplib.h` under `-Werror`)

```sh
just test       # unit tests, token check, pytest, installer tests
just lint       # gates; clang-tidy in work/venv-tidy (tools/gates/requirements-tidy-macos.txt)
```

Metal backend (needs `xcodebuild -downloadComponent MetalToolchain`):

```sh
cmake -S engine -B work/build/mtl -G Ninja -DTONA_BACKEND_METAL=ON -DCMAKE_BUILD_TYPE=Release \
    -DOPENSSL_ROOT_DIR=$(brew --prefix openssl@3)
ninja -C work/build/mtl tona-check tona-server
work/build/mtl/tona-check MODEL.gguf "PROMPT" -n 200   # against the CPU reference
```

`just build` builds in `work/build/engine`; `tona start` runs the server in the background (log `work/tona-server.log`)
until `tona stop`.

CI: job `engine-macos` in `.github/workflows/tests.yml` (`macos-15`) builds with Metal, runs `ctest -L unit` and the
installer tests, no model ([backends/README.md](../backends/README.md#tiers)). The release assets come from `scripts/release-build-macos.sh`.

## What differs from Linux

| Part | Linux (reference) | macOS |
|---|---|---|
| CPU target | `-march=${TONA_MARCH}` (znver5) | `-mcpu=${TONA_MCPU}` (apple-m1) |
| Row reads (`DirectRows`, PLE table) | io_uring, `io/uring.cpp` | one `pread` per row, `io/uring_pread.cpp`; `evict()` does nothing (use `direct`, i.e. `F_NOCACHE`) |
| CPU experts | AVX-512 | not built; `cpu/experts_unsupported.cpp` throws |
| RAM accounting (`backend/budget.h`) | `/proc`, cgroups | Mach statistics (`backend/mem_macos.h`) |
| Page-cache drop of mapped weights | `madvise`, `posix_fadvise` | `madvise` only |
| Updater restart | `systemctl --user restart tona-server` | ends the `tona-server` of `server.pid`, starts `tona start --foreground` (`update/script_service.cpp`) |
| Server start | systemd user unit that runs `tona start --foreground` | `tona start --foreground` in the background; at login only with `tona autostart on` (LaunchAgent runs the same command) |
| `scripts/gpu-status.sh` tests | run | skipped |
