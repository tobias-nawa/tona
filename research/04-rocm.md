# ROCm 10 for gfx1200

Agent research of 2026-10-22 (session in the fedora repo). Release notes read,
AMD's own package sources queried.

> **Decided on 2026-10-22: route B**, stable ROCm 10.1.0 wheels in `work/rocm-10.1`
> (`scripts/setup.sh`, `source toolchain/env.sh`, commit bed72c0). The findings from
> setting it up are below under "Setting up route B".

## Versions

| Version | Date | Status |
|---|---|---|
| 7.1.1 | – | **installed locally** (Fedora package `rocm-runtime-7.1.1-6.fc44`), used by a local inference service |
| 7.14 | 2026 | from here on AMD builds ROCm with **TheRock** |
| 10.0.0 | 2026-08-26 | a "test release" according to the release notes |
| **10.1.0** | **2026-10-05** | current version |
| 10.2 | nightlies | `10.2.0a20260930` is in use by other projects under Windows |

The release notes do not say why AMD jumps from 7.14 to 10.0. The components carry their
own numbers: 10.1.0 contains HIP 7.16.0 and LLVM 24.0.0 (since 10.0.0).

## What 10.x brings for us

| Area | New | Meaning for the project |
|---|---|---|
| **ROCm Compute Profiler** | interactive **roofline report** (10.1) | shows the distance to the hardware limit per kernel; exactly our metric |
| **ROCprofiler-SDK** | kernel replay (beta, 10.1), HIP graph mapping per node (10.0), HIP event tracing (10.1) | repeatable kernel measurement, profiling of graphs |
| Compiler | LLVM 23 → **24** (10.0), HIPIFY 10.0.0 | newer codegen for gfx12; HIPIFY helps port CUDA paper code |
| HIP runtime | 7.15 (10.0) → 7.16 (10.1): faster `hipEventRecord` with `hipEventDisableTiming`, cooperative-groups scans, `hipMemGetDefaultMemPool` | smaller building blocks for synchronization and memory pools |
| Composable Kernel | 1.2.0 → 1.3.0: FMHA forward for hdim 128 faster on gfx11/gfx12 | reference for our attention kernels |
| hipBLASLt | 1.4.1 (unchanged since 10.0) | tuning tables for gfx1200 not mentioned |
| rocWMMA | 2.2.1, unchanged | – |
| rocSPARSE | wave32 tuning for SpMV/SpMM on gfx1201 (10.1) | gfx1201, not gfx1200 |
| MIOpen | gfx1201 SystemDB updated (10.0), OpenCL backend removed | hardly relevant for us |
| Frameworks | vLLM 0.29.0, SGLang 0.5.18, JAX 0.11.1 list gfx1200/gfx1201 (10.1) | basis for comparison; PyTorch 2.12/2.13 Instinct only |

**No new hardware capabilities for gfx1200 itself.** New in 10.0 are only the RX 9050
cards (also gfx1200). FP8 on RDNA4 is not mentioned in the release notes.

Breaking changes in 10.0, in case we use libraries directly: AMD SMI 27.0.0 (new
SONAME `libamd_smi.so.27`), rocSPARSE without `rocsparse_indextype_u16`, MIOpen without
the OpenCL backend.

## Platform support

Fedora is **not** supported. For Radeon, 10.1.0 lists: Ubuntu 26.04.1, 24.04.5,
22.04.5, RHEL 10.2 and 9.8, Windows 11 25H2.

## Ways to obtain it (queried ourselves on 2026-10-22)

| Source | What is there |
|---|---|
| `repo.amd.com/rocm/tarball/` (stable, per GPU family) | `therock-dist-linux-gfx120X-all-7.10.0` to **`7.13.0`**, no 7.14 or 10.x |
| `repo.amd.com/rocm/whl/gfx120X-all/rocm/` (stable, wheels) | `rocm-7.10.0` to **`7.13.0`** |
| `rocm.nightlies.amd.com/v2/gfx120X-all/rocm/` (nightly, per family) | latest `rocm-7.14.0a20260612` |
| **`stable.repo.amd.com/rocm/whl-next/`** (stable, multi-arch; redirects to `…/rocm/core/whl-next/`) | **`rocm-10.0.0`, `rocm-10.1.0`**; plus `rocm-sdk-core`, `-libraries`, `-devel`, `rocm-sdk-device-gfx1200` (`10.1.0-py3-none-linux_x86_64`), `rocm-profiler` |
| `rocm.nightlies.amd.com/whl-multi-arch/` (nightly, multi-arch) | `rocm-10.1.0a202608xx`, plus `rocm-sdk-device-gfx1200` |
| `rocm.nightlies.amd.com/tarball-multi-arch/` | `therock-dist-linux-multiarch-tests-10.1.0a202608xx` |

Distribution has moved to **multi-arch packages** since 7.14; the family channels end at
7.13 and the 7.14 nightlies. **ROCm 10.1.0 exists as stable wheels** at
`stable.repo.amd.com/rocm/whl-next/`; the first version of this file wrongly described
route B as a nightly-only route.

## Options

| | Route | Advantage | Drawback |
|---|---|---|---|
| **A** | Podman container, Ubuntu 26.04 + official ROCm 10.1, `--device /dev/kfd --device /dev/dri` | officially supported combination, reproducible, Fedora's 7.1.1 for the local inference service stays untouched | development through the container; set up SELinux and device access (see below) |
| **B** | stable Python wheels from `stable.repo.amd.com/rocm/whl-next/` in a venv (`work/rocm-10.1`) | no sudo, stable 10.1.0 | not officially supported on Fedora; Fedora's headers can interfere (see below) |
| C | tarball to `/opt` | system-wide | stable for gfx120X only up to 7.13, needs sudo |

The fedora session had recommended A as long as it took B for a nightly-only route.
**Decision: B.** A stays documented as a fallback; nothing is prepared for it in the
fedora repo.

## Setting up route B (inference-engine session, 2026-10-22)

- 10.1.0 wheels, 6.9 GB in `work/rocm-10.1`, pinned with SHA-256 in
  `toolchain/rocm-10.1.lock`, set up by `scripts/setup.sh`, no sudo.
- `rocm-sdk test`: 31/31 passed (only with the environment activated; otherwise Fedora's
  `/usr/bin/hipconfig` takes over).
- `rocminfo`: gfx1200 RX 9060 XT. HIP 7.16.26385, AMD clang 24.0.0.
- WMMA f16→f32 16×16×16 against a CPU reference: correct.
- rocprofv3: kernel trace and hardware counters (`SQ_WAVES`, `GRBM_GUI_ACTIVE`) work with
  `perf_event_paranoid=2`, **without a system change**. `rocprof-compute` installed, not
  yet tested.
- **Pitfall:** Fedora's `/usr/include/hip` (HIP 7.1) shadows the 10.1 headers, even with
  `amdclang++ --rocm-path`. Fix in `toolchain/env.sh`:
  `-isystem $ROCM_PATH/include`. The smoke test compares header and runtime version and
  aborts on a mix.
- Open: microbenchmarks 7.1.1 versus 10.1, test `rocprof-compute`.

## System state for option A (checked 2026-10-22, reference only)

| Item | Finding |
|---|---|
| Device files | `/dev/kfd`, `/dev/dri/renderD128`, `renderD129` with `crw-rw-rw-`, group `render`. Access works even **without** membership in `render`/`video` |
| Groups | The user is in `wheel cdrom libvirt`, **not** in `render`/`video`. `--group-add keep-groups` therefore achieves nothing, but is not needed at 0666 either |
| SELinux | `Enforcing`, boolean `container_use_devices` **off** |
| Podman | 5.8.9 |
| Perf counters | `kernel.perf_event_paranoid = 2`. On the host this is enough for rocprofv3 including hardware counters (see route B); unchecked in the container |
| Local inference service | `enabled`, `WantedBy=graphical.target`, `Restart=always`, no other services depend on it. Currently holds no model in VRAM |

SELinux, two ways:

- `--security-opt label=disable` only for the development container: no system change,
  affects only that one container, which then runs without SELinux separation.
- `setsebool -P container_use_devices on`: permanent system change, allows **all**
  containers device access.

The local inference service for measurement runs: `systemctl stop <service>` ends the service cleanly,
`Restart=always` does not apply to a manual stop. After the run `systemctl start <service>`,
otherwise it only starts at the next boot. In the meantime local models do not work in
opencode.

## Sources

- [ROCm 10.1.0 release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
- [ROCm 10.0.0 release notes](https://rocm.docs.amd.com/en/docs-10.0.0/about/release-notes.html)
- [ROCm 7.14.0 release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html)
- [TheRock: RELEASES.md](https://github.com/rocm/therock/blob/main/RELEASES.md)
- [TheRock: SUPPORTED_GPUS.md](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
- Package sources: `https://stable.repo.amd.com/rocm/whl-next/`, `https://repo.amd.com/rocm/tarball/`, `https://repo.amd.com/rocm/whl/`, `https://rocm.nightlies.amd.com/`
- Route B setup: commit bed72c0
