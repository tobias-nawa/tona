"""The release check of the needed libraries: a library that is neither shipped nor of the base system is reported."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import release_needed

READELF = """
Dynamic section at offset 0x1234 contains 20 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libamdhip64.so.7]
 0x0000000000000001 (NEEDED)             Shared library: [libgomp.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN/../lib]
"""


def test_needed_libraries_are_read_in_order():
    expected = ["libamdhip64.so.7", "libgomp.so.1", "libstdc++.so.6", "libc.so.6"]
    assert release_needed.needed_libraries(READELF) == expected


def test_shipped_and_base_system_libraries_pass():
    needed = release_needed.needed_libraries(READELF)
    assert release_needed.missing(needed, {"libamdhip64.so.7", "libgomp.so.1"}) == []


def test_the_openmp_runtime_forgotten_by_the_release_is_reported():
    needed = release_needed.needed_libraries(READELF)
    assert release_needed.missing(needed, {"libamdhip64.so.7"}) == ["libgomp.so.1"]


def test_a_runpath_line_is_not_a_needed_library():
    assert release_needed.needed_libraries(" 0x000000000000001d (RUNPATH) Library runpath: [libx.so]") == []


def test_origin_relative_runpath_passes():
    assert release_needed.bad_runpath(release_needed.runpath_entries(READELF)) == []


def test_a_toolchain_path_in_the_runpath_is_reported():
    leaked = READELF + (
        " 0x000000000000001d (RUNPATH)            Library runpath: "
        "[$ORIGIN/../lib:/home/u/work/rocm-10.1/lib/llvm/lib:/home/u/work/rocm-10.1/lib]\n"
    )
    entries = release_needed.runpath_entries(leaked)
    assert release_needed.bad_runpath(entries) == ["/home/u/work/rocm-10.1/lib/llvm/lib", "/home/u/work/rocm-10.1/lib"]


def test_old_style_rpath_is_checked_too():
    line = " 0x000000000000000f (RPATH)              Library rpath: [/opt/x/lib]"
    assert release_needed.bad_runpath(release_needed.runpath_entries(line)) == ["/opt/x/lib"]


def test_no_runpath_is_fine():
    assert release_needed.bad_runpath(release_needed.runpath_entries(" 0x1 (NEEDED) Shared library: [libc.so.6]")) == []
