# Licences of the shipped ROCm and LLVM libraries (2026-10-28)

Agent research (Sonnet), spot-checked by hand: the LLVM licence file in the wheels has 6 lines, NOTICES.txt holds GPL
texts of software we do not ship, glog and libhsakmt strings are present in the binaries.


Date: 2026-10-28. Scope: libamdhip64.so.7, libhsa-runtime64.so.1, libamd_comgr.so.3, libLLVM.so.24.0git,
libclang-cpp.so.24.0git, librocprofiler-register.so.0, librocm_kpack.so.0 (rocm_sysdeps already handled).
Nothing in the repo was changed. Legal reading = what the licence texts literally say; this is not legal advice.

Legend: [LOCAL] = checked on this machine (file + line). [WEB] = fetched from a URL on 2026-10-28.
[NOT VERIFIED] = could not be confirmed.

Workspace of the checks: /var/tmp/claude/lic (sparse checkouts of ROCm/llvm-project and ROCm/rocm-systems at the pins).

## 0. Source pins (which source the binaries come from)

- [WEB] TheRock tag `therock-10.1`: submodule `compiler/amd-llvm` = ROCm/llvm-project commit
  `d6f6cb691863e9258034ab48c1dfce21689d5606`
  (`gh api repos/ROCm/TheRock/contents/compiler?ref=therock-10.1`, entry `amd-llvm`).
- [LOCAL] The binaries confirm it: `strings libLLVM.so.24.0git` -> `24.0.0git d6f6cb691863e9258034ab48c1dfce21689d5606`;
  `strings libamdhip64.so.7` -> `AMD clang version 24.0.0git (https://github.com/ROCm/llvm-project.git d6f6cb69...)`.
- [WEB] TheRock `therock-10.1`: submodule `rocm-systems` = ROCm/rocm-systems commit
  `7da43026e4f2ef7600433c38c820732f7a205d22` (same commit that licenses/rocm-sysdeps.txt already names for kpack).
  hip/clr -> `projects/clr`, `projects/hip`; hsa -> `projects/rocr-runtime`; `projects/rocprofiler-register`; `shared/kpack`.
- comgr lives in the LLVM repo: `amd/comgr` at d6f6cb69.

## 1. LLVM / Clang (libLLVM, libclang-cpp)

### 1.1 Licence
- [WEB] `llvm/LICENSE.TXT` and `clang/LICENSE.TXT` at d6f6cb69 (the two differ only in the legacy copyright year line,
  2003-2019 vs 2007-2019): "Apache License v2.0 with LLVM Exceptions" = full Apache-2.0 text (lines 1-~203),
  then "---- LLVM Exceptions to the Apache 2.0 License ----", then "Software from third parties included in the LLVM
  Project", then "Legacy LLVM License" = University of Illinois/NCSA (Copyright (c) 2003-2019 University of Illinois at
  Urbana-Champaign). Source files carry `SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception`.
  URL: https://raw.githubusercontent.com/ROCm/llvm-project/d6f6cb691863e9258034ab48c1dfce21689d5606/llvm/LICENSE.TXT
  (sha256 8d85c1057d742e597985c7d4e6320b015a9139385cff4cbae06ffc0ebe89afee, 279 lines).
- The legacy NCSA licence applies only to code contributed before the 2019 relicensing and is dual-offered; the
  file lists it, so shipping the whole LICENSE.TXT covers both.

### 1.2 What Apache-2.0 section 4 demands for an unmodified binary (verbatim, llvm/LICENSE.TXT lines 93-126)
"You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without
modifications, and in Source or Object form, provided that You meet the following conditions:
(a) You must give any other recipients of the Work or Derivative Works a copy of this License; and
(b) You must cause any modified files to carry prominent notices stating that You changed the files; and
(c) You must retain, in the Source form of any Derivative Works that You distribute, all copyright, patent, trademark,
and attribution notices ...; and
(d) If the Work includes a "NOTICE" text file as part of its distribution, then any Derivative Works that You distribute
must include a readable copy of the attribution notices contained within such NOTICE file ..."
Consequences for us:
- (a) applies: a copy of the Apache-2.0 text must go along. The comgr LICENSE.txt in the wheel contains the full
  Apache text, so (a) is textually met today; but the LLVM project's own copyright/legacy notice is not in the file (see 3).
- (b) We do not modify the files (AMD's build is unmodified by us). AMD's fork has changes, but those are AMD's duty
  upstream of us; nothing to add. (Reading of the text; not verified with a lawyer.)
- (c) only concerns Source form. We ship binaries only. No obligation.
- (d) only if a NOTICE file exists. Section 1.4: there is none.
- Section 6 (lines 142-145): "This License does not grant permission to use the trade names, trademarks, service
  marks, or product names of the Licensor, except as required for reasonable and customary use in describing the
  origin of the Work". -> say "AMD ROCm" only descriptively.
- No obligation to offer source code (Apache-2.0 has no such clause; text of section 4 above). Source pointer is
  courtesy only.

### 1.3 What the LLVM Exception covers
Verbatim (LICENSE.TXT lines ~205-218): "if, as a result of your compiling your source code, portions of this Software
are embedded into an Object form of such source code, you may redistribute such embedded portions in such Object form
without complying with the conditions of Sections 4(a), 4(b) and 4(d) of the License." plus a GPLv2 patent/indemnity
waiver option.
- It is aimed at code that is compiled WITH LLVM and embeds LLVM runtime parts (compiler-rt, libc++ ...). It does
  NOT exempt redistributing libLLVM.so / libclang-cpp.so themselves: those are the Work in Object form, not "portions
  embedded into an Object form of your source code". So sections 4(a) (licence copy) fully applies to us.
  (Own reading of the text, [NOT VERIFIED] by a lawyer or LLVM FAQ.)
- It does not touch trademarks (section 6), nor the third-party licences in 1.5.

### 1.4 NOTICE file
- [WEB] No NOTICE file in ROCm/llvm-project at d6f6cb69: `llvm/NOTICE.TXT`, `NOTICE.TXT`, `NOTICE`, `llvm/NOTICE`,
  `clang/NOTICE.TXT`, `amd/comgr/NOTICE` all return HTTP 404. Tree listing of `llvm/` and `clang/` (git trees API)
  shows no NOTICE*. So Apache 4(d) is empty.
- [LOCAL] Wheels contain no NOTICE for LLVM either (section 3).

### 1.5 Third-party code compiled into libLLVM / libclang-cpp
Method [LOCAL]: sparse checkout of llvm/lib, clang/lib at d6f6cb69; listed every .c/.cpp/.h/.inc without the standard
"Part of the LLVM Project" header (115 files) and grepped headers for BSD/MIT/PD/Unicode terms. Result for code that
actually lands in the two libraries (llvm/lib/Support; clang/lib has none besides test-like headers):

| Component | Files (llvm/lib/Support/...) | Licence | Text needed? |
|---|---|---|---|
| xxHash (Yann Collet, 2012-2023) | xxhash.cpp lines 1-33 | BSD-2-Clause | yes, binary redistribution must reproduce notice |
| MD5 (Alexander Peslyak) | MD5.cpp lines 1-37 | public domain / "Redistribution and use ... permitted" (cut-down BSD) | no text required; include for completeness |
| OpenBSD regex (Henry Spencer + UC Berkeley) | regcomp.c, regexec.c, regerror.c, regfree.c, regstrlcpy.c, regex2.h, regex_impl.h, regutils.h, regengine.inc; `COPYRIGHT.regex` | Spencer licence (credits must appear in documentation) + BSD-3-Clause (Berkeley) | yes: COPYRIGHT.regex, regcomp.c lines 1-36 |
| Unicode ConvertUTF (Unicode, Inc. 1991-2015) | ConvertUTF.cpp lines 8-44 | Unicode licence (copyright+permission notice must appear with copies or in documentation) | yes |
| Unicode name table (Unicode, Inc. 1991-2022) | UnicodeNameToCodepointGenerated.cpp lines 16-63 | Unicode, Inc. licence agreement | yes |
| BLAKE3 (C impl.) | BLAKE3/LICENSE (282 lines), blake3.c `SPDX: CC0-1.0` | CC0-1.0 or Apache-2.0 (choose Apache; Apache text already present) | no extra text (Apache chosen); optional |
| SHA1 | SHA1.cpp l.9 "taken from public domain" (NetBSD) | public domain | no |
| SHA256 | SHA256.cpp "nacl ... public domain" | public domain | no |
| rpmalloc 1.4.5 fork | rpmalloc/README.md | public domain or MIT | no (public-domain option) |
| SipHash | third-party/siphash/include/siphash/SipHash.h: LLVM header, "Lightly adapted from the SipHash reference C implementation" | Apache-2.0 WITH LLVM-exception (LLVM header) | no |

Clang: only clang/lib/Headers/*.h (MIT-style Intel headers etc.) and rocclr's GL headers carry other licences; those are
resource headers, not part of libclang-cpp.so. [NOT VERIFIED at object level: I did not disassemble which Support
objects the linker kept; libLLVM links the whole llvm/lib/Support, so assume all of the above are in.]
Dynamically linked deps of both libs: zlib, zstd (rocm_sysdeps, already covered), libstdc++/libgcc/glibc (system).

## 2. AMD libraries

| Library | Source (pin) | Licence | Evidence |
|---|---|---|---|
| libamdhip64.so.7 | rocm-systems projects/clr (hipamd, rocclr, opencl not shipped), projects/hip | MIT, "Copyright (C) Advanced Micro Devices, Inc." | clr/LICENSE.md, hipamd/LICENSE.md, rocclr/LICENSE.md, hip/LICENSE.md all md5 7c1914fa... = wheel `share/doc/hip/LICENSE.md` [LOCAL+WEB] |
| libhsa-runtime64.so.1 | projects/rocr-runtime/runtime/hsa-runtime + libhsakmt (static) | University of Illinois/NCSA (AMD) | LICENSE.md md5 46b51815... identical to wheel `rocr/LICENSE.md`; libhsakmt/LICENSE.md is MIT + BSD-2 |
| libamd_comgr.so.3 | llvm-project amd/comgr | Apache-2.0 WITH LLVM-exception + "Legacy Comgr License" (NCSA, 2018-2025 AMD) | amd/comgr/LICENSE.txt md5 02e02ae0... identical to wheel `amd_comgr/LICENSE.txt` |
| librocprofiler-register.so.0 | projects/rocprofiler-register | MIT, AMD | LICENSE.md md5 7c1914fa... identical to wheel |
| librocm_kpack.so.0 | shared/kpack @7da4302 | MIT | already in licenses/rocm-sysdeps.txt |

ROCm's own overview page confirms per-component licences: https://rocm.docs.amd.com/en/latest/about/license.html
(fetched 2026-10-28): CLR MIT, HIP MIT, hipamd MIT, Comgr NCSA, ROCR-Runtime NCSA, ROCm-Device-Libs NCSA,
rocminfo NCSA, llvm-project Apache, ROCprofiler-SDK MIT. (The page says Comgr is NCSA; the file at our pin is Apache+LLVM
exception with NCSA as "legacy". The file in the pin and in the wheel wins.)

Obligations per licence text:
- MIT (hip, rocprofiler-register, kpack, libhsakmt MIT part): "The above copyright notice and this permission notice
  shall be included in all copies or substantial portions of the Software." -> include the text.
- NCSA (rocr, comgr legacy, device-libs): "Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimers in the documentation and/or other materials provided with the
  distribution." and "Neither the names of Advanced Micro Devices, Inc, nor the names of its contributors may be used
  to endorse or promote products derived from this Software without specific prior written permission." (rocr/LICENSE.md)
  -> include text; no endorsement use of AMD's name.
- Apache-2.0 WITH LLVM-exception (comgr main licence): as 1.2.

### 2.1 Additional third-party code inside the AMD libraries (found by me; not in the wheels' licence files)
All [LOCAL] evidence = strings/nm of the shipped binary, [WEB]/[LOCAL] = source at the pins.
1. **libhsakmt statically linked into libhsa-runtime64.so.1.** `strings` shows 452 `hsaKmt`/`/dev/kfd` hits, no NEEDED
   libhsakmt, no exported hsaKmt symbols; the wheel ships only `libhsakmt.a`. Licence file
   `projects/rocr-runtime/libhsakmt/LICENSE.md` = AMD MIT text + "This product contains software provided by Nginx, Inc.
   ... Copyright (C) 2002-2018 Igor Sysoev, (C) 2011-2018 Nginx, Inc." with a BSD-2-Clause text (rbtree.c/rbtree.h).
   File headers also name Red Hat (virtgpu_drm.h, MIT) and Raptor Engineering (topology.c, MIT, "notice ... shall be
   included in all copies or substantial portions"). `grep -r 'Igor Sysoev'` over all wheels finds only rocdxg, not
   libhsakmt: it is NOT in the wheel licence files and NOT in licenses/rocm-sysdeps.txt (0 hits).
   URL: https://raw.githubusercontent.com/ROCm/rocm-systems/7da43026e4f2ef7600433c38c820732f7a205d22/projects/rocr-runtime/libhsakmt/LICENSE.md
   (sha256 f45a5bc0bc68dd9a00e9609097788b1cbda79726fee02549cb0c04b90190b419).
2. **ELFIO (Serge Lamikhov-Center, MIT) inside libamdhip64.so.7.** Symbols `N3amd5ELFIO...` in the binary [LOCAL];
   source `projects/clr/rocclr/elf/elfio/*.hpp`, header "Copyright (C) 2001-2015 by Serge Lamikhov-Center ... The above
   copyright notice and this permission notice shall be included in all copies or substantial portions of the Software."
   Notice = first 22 lines of elfio.hpp.
   URL: https://raw.githubusercontent.com/ROCm/rocm-systems/7da43026e4f2ef7600433c38c820732f7a205d22/projects/clr/rocclr/elf/elfio/elfio.hpp (lines 1-22). Not in wheel licence files.
3. **glog (Google, BSD-3-Clause) and fmt (MIT) statically linked into librocprofiler-register.so.0.** `strings`: 379
   `GLOG_*`/`N6google...` hits and 143 `fmt::v`/`N3fmt` hits [LOCAL]; submodules in rocprofiler-register/external:
   glog `7b134a5c82c0c0b5698bb6bf7a835b230c5638e4`, fmt `123913715afeb8a437e6388b4473fcc4753e1c9a` (git tree at 7da4302).
   - glog COPYING (28 lines, "Copyright (c) 2024, Google Inc.", BSD-3): https://raw.githubusercontent.com/google/glog/7b134a5c82c0c0b5698bb6bf7a835b230c5638e4/COPYING
     sha256 136d48dea7a681413691f3db3098f6cf5ffaa3119d96d97bb83b8cff3ce38c4a. BSD-3 binary clause: reproduce notice.
   - fmt LICENSE (27 lines, MIT + "Optional exception": embedded portions in a machine-executable object form may be
     redistributed without notices): https://raw.githubusercontent.com/fmtlib/fmt/123913715afeb8a437e6388b4473fcc4753e1c9a/LICENSE
     sha256 07580f2a3b35709ce703d523f447b242f6dfec7582a8c0df102c7fa2849375f8. Exception probably covers us; include anyway (cheap).
   Not in wheel licence files.
4. **ROCm-Device-Libs (NCSA) bitcode in libamdhip64.so.7.** `strings` shows `__asan_*` and `__ockl_dm_init_v1`,
   `__ockl_sanitizer_report` in embedded bitcode; those functions are defined in llvm-project `amd/device-libs`
   (asanrtl/src/dm.cl) [LOCAL source grep]. Licence: `amd/device-libs/LICENSE.TXT` ("ROCm-Device-Libs Release License",
   NCSA, 43 lines; binary clause as above). The wheel carries no copy of it (find over all wheels: no device-libs licence).
   URL: https://raw.githubusercontent.com/ROCm/llvm-project/d6f6cb691863e9258034ab48c1dfce21689d5606/amd/device-libs/LICENSE.TXT
   sha256 a4e8e3ba6e9255928a29d3085f929de402e61413cb0c49276e16b558effa8a2a.
   [NOT VERIFIED]: that the exact embedded object code stems from device-libs rather than the HIP-owned copy; treat as yes (cost: 43 lines).
5. rocclr/include/GL/glx.h, glxext.h (Brian Paul, Khronos): header-only declarations, not code in the binary. Ignore.

## 3. What the wheels already carry

[LOCAL] `find work/rocm-10.1/lib/python3.14/site-packages ... -iname 'LICENSE*' -o -iname 'NOTICE*' -o -iname 'COPYING*'`:
- `_rocm_sdk_core/share/doc/{amd_comgr,hipcc,hsa-amd-aqlprofile,rocminfo,rocprofiler-register,rocr,roctracer,roctracer-asan}/` licence files.
- `_rocm_sdk_devel/share/doc/` the same plus ~45 more components (hip/LICENSE.md, rocm-core, half, rocblas ...) and the single
  `_rocm_sdk_devel/share/doc/NOTICES.txt`. NOTE: `hip/LICENSE.md` exists only in the devel wheel, not in core.
- `_rocm_sdk_devel/lib/llvm/include/llvm/Support/LICENSE.TXT` (6 lines).
- `*.dist-info`: only METADATA/WHEEL/RECORD, no licence field and no `licenses/` folder for the rocm wheels (only pip's).
- No LLVM LICENSE.TXT/NOTICE, no device-libs licence, no libhsakmt licence, no ELFIO/glog/fmt text anywhere
  (`find` for LICENSE/NOTICE/COPYING with llvm/clang in path -> only the 6-line file).

Does AMD's NOTICES.txt cover LLVM? **No.** [LOCAL] `_rocm_sdk_devel/share/doc/NOTICES.txt` (808 lines, 40 KB): content =
header "Notices and licenses file"; "AMD copyrighted code (GPLv3)" (GDB); "AMD copyrighted code (UoI)" (NCSA, "Copyright (c)
2018 AMD"); gnu-gdb, gnu-bash, gnu-glibc (LGPL2.1 header), full GPLv3 text, and "zlib 1.2.11". `grep -i 'llvm\|apache\|blake\|xxhash\|unicode'`
-> only generic matches of the word "notice"; zero LLVM, Apache, Unicode, xxhash, regex, BLAKE3 content. It is the notices
file for rocgdb/devel tooling, not for the runtime libs. Shipping it adds 40 KB of GPLv3/bash/glibc text for software we
do NOT ship (confusing; could be read as claiming we ship GPL code).

The "LLVM" file the script includes (`lib/llvm/include/llvm/Support/LICENSE.TXT`) is the 6-line header of the
"LLVM System Interface Library" directory:
"Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions. See https://llvm.org/LICENSE.txt ...".
It is NOT the LLVM licence text. So 4(a) for libLLVM/libclang-cpp is met only indirectly through comgr's LICENSE.txt
(which has the Apache text but under the comgr heading and without the LLVM-project legacy notice).

Check of the wheel files against upstream: hip/rocprofiler-register/rocr/comgr licence files are byte-identical (md5)
to the files at the pinned commits [LOCAL+WEB]; so the wheel files themselves are correct, they are just incomplete.

## 4. Gaps in the licence file release-build.sh produces today

Current content (scripts/release-build.sh lines 102-118): licenses/rocm-sysdeps.txt (1313 lines: zlib, zstd, xz/liblzma,
bzip2, libelf, libnuma, libdrm, kpack) + NOTICES.txt + hip/LICENSE.md + amd_comgr/LICENSE.txt + rocr/LICENSE.md +
rocprofiler-register/LICENSE.md + llvm/Support/LICENSE.TXT.

Missing (in priority order), all fetchable with curl, pinned:
1. **Real LLVM licence**, replacing the 6-line stub:
   https://raw.githubusercontent.com/ROCm/llvm-project/d6f6cb691863e9258034ab48c1dfce21689d5606/llvm/LICENSE.TXT
   (sha256 8d85c105...afee). Covers libLLVM, libclang-cpp, comgr (Apache + exception + legacy UIUC).
2. **LLVM third-party notices** (not covered by any shipped text):
   - xxHash BSD-2: xxhash.cpp lines 1-33 at the pin (…/llvm/lib/Support/xxhash.cpp)
   - OpenBSD regex: …/llvm/lib/Support/COPYRIGHT.regex (sha256 0424e57d…c97c7) and the Berkeley BSD-3 block in regcomp.c lines 1-36
   - Unicode: …/llvm/lib/Support/ConvertUTF.cpp lines 8-44 and UnicodeNameToCodepointGenerated.cpp lines 16-63
   - MD5 (public domain notice): MD5.cpp lines 1-37 (optional)
   - BLAKE3: Apache option chosen, so nothing more; name it in the list (…/BLAKE3/LICENSE sha256 6a94bedb…c0e)
   All with prefix https://raw.githubusercontent.com/ROCm/llvm-project/d6f6cb691863e9258034ab48c1dfce21689d5606/
3. **libhsakmt** LICENSE.md (MIT + Nginx BSD-2), see 2.1(1); copy it and name Red Hat/Raptor MIT headers.
4. **ELFIO MIT** notice (2.1(2)).
5. **glog BSD-3** and **fmt MIT** (2.1(3)), librocprofiler-register.
6. **ROCm-Device-Libs NCSA** (2.1(4)).
7. Cleanup (not a gap, a defect): drop AMD NOTICES.txt (GPL/bash/glibc text for non-shipped rocgdb; only the 2018 NCSA block and
   zlib 1.2.11 in it matter, and zlib 1.3.2 is already in rocm-sysdeps.txt); drop the 6-line LLVM stub.
   Also hip/LICENSE.md is read from the devel `share/doc`; fine as long as `$ROCM_PATH` is the merged venv.
8. A short header section in the file: for each shipped library -> component -> licence -> pinned source URL (so a reader
   can map libraries to texts). The existing rocm-sysdeps.txt has that style per entry.

## 5. Other points relevant for redistribution

- **Source offer:** only LGPL (elfutils libelf, libnuma), already in docs/maintain/release.md. Apache-2.0, MIT, NCSA, BSD, Unicode
  require no source offer (text of section 4 above; MIT/BSD/NCSA texts only demand notices). The Apache FAQ page was fetched but gave nothing quotable
  on this point [NOT VERIFIED by FAQ]; the licence text itself is the evidence.
- **Endorsement/naming:** NCSA texts (rocr, comgr legacy, device-libs, LLVM legacy) and BSD-3 (glog, Berkeley regex): "Neither the
  names of ... may be used to endorse or promote products derived from this Software without specific prior written
  permission." Apache s.6 same for trademarks. Practical: write "uses AMD ROCm 10.1 runtime libraries (unmodified)", no AMD/LLVM logo,
  no claim of endorsement. Also mention "unmodified binaries from AMD's wheels" - fine under all texts.
- **Trademarks AMD / ROCm / Radeon:** AMD legal trademark page could not be fetched (HTTP 000/blocked; I did not work around it).
  [NOT VERIFIED]. Standard practice: attribution line "AMD, Radeon, ROCm are trademarks of Advanced Micro Devices, Inc." in docs/README.
- **Export control:** [NOT VERIFIED]. No AMD statement found for these wheels. Publicly available open-source software is generally
  outside the US EAR (15 CFR 734.7); the libs contain no encryption beyond hashing (BLAKE3/SHA/MD5). Not legal advice; no obligation in the
  licence texts.
- **Wheel/index terms:** the AMD wheel metadata has no licence field; the pip index `stable.repo.amd.com/rocm/whl-next/` terms of use were not
  looked up [NOT VERIFIED]. Per-component licences above are what the files say.
- **Patents:** Apache-2.0 s.3 grants a patent licence with termination if you sue; nothing for us to do.
- **Modifications:** we ship the bytes of the wheels unchanged (SHA-256 pinned in toolchain/rocm-10.1.lock); keep it so. If a library is ever patched
  or rebuilt, Apache 4(b) (change notice) starts to apply.

## 6. To-do for scripts/release-build.sh (and licenses/)

1. Vendor into `licenses/` (checked-in, hash-checked in the script like the lock file), not fetched at build time:
   - `licenses/llvm-LICENSE.TXT` (llvm/LICENSE.TXT @ d6f6cb69, sha256 8d85c105...afee)
   - `licenses/llvm-third-party.txt`: one file with the extracted texts from 2 (xxhash header, regcomp.c Berkeley/Spencer header +
     COPYRIGHT.regex, ConvertUTF.cpp Unicode block, UnicodeNameToCodepointGenerated.cpp Unicode block, MD5.cpp block, BLAKE3 note), each with source URL @ pin.
   - `licenses/rocm-extra.txt` (or extend rocm-sysdeps.txt): libhsakmt LICENSE.md (+ Red Hat, Raptor MIT notes), ELFIO MIT notice,
     glog COPYING, fmt LICENSE, ROCm-Device-Libs LICENSE.TXT, each with "linked into <library>" and pinned URL.
2. In the licence block (lines ~110-115): remove `$docs/NOTICES.txt` and `$ROCM_PATH/lib/llvm/include/llvm/Support/LICENSE.TXT`; add
   the new vendored files; keep hip, amd_comgr, rocr, rocprofiler-register from the wheel.
3. Put a header table at the top: library -> licence(s) -> files in the text below. Update the heading line ("wheels carry licence files only for ..."
   comment on line 102-103 is wrong: they have none for LLVM proper, libhsakmt, ELFIO, glog, fmt, device-libs).
4. Add a test (tests for release-build/manifest already exist, see commits for #67) that the generated LICENSES.txt contains: "Apache License",
   "LLVM Exceptions", "University of Illinois/NCSA", "Igor Sysoev", "Serge Lamikhov-Center", "Google Inc.", "fmt" exception text,
   "Henry Spencer", "Unicode, Inc.", "Yann Collet", and does not contain "GNU GENERAL PUBLIC LICENSE Version 3" (guard against NOTICES.txt returning).
5. docs/maintain/release.md table row for `LICENSES.txt`: update the description (LLVM, third-party and statically linked components named).
6. Add the trademark attribution sentence to the README/release notes (AMD, Radeon, ROCm). Say "unmodified AMD binaries".
7. Optional: record the pins (llvm-project d6f6cb69, rocm-systems 7da4302, glog 7b134a5c, fmt 12391371) in licenses/ header so the next ROCm bump
   knows what to re-check (re-run the header scan: `grep -L 'Part of the LLVM Project'` over llvm/lib, `strings | grep GLOG_`, `nm`/`strings` for
   statically linked third-party in hsa/hip/register).
8. On every ROCm version bump: recheck pins (TheRock submodules), diff licence files, and re-run the `strings`/`ldd` static-link check
   (new static deps appear without notice, e.g. glog/fmt/ELFIO/hsakmt were not announced anywhere).
