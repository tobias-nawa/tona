import shutil
import subprocess

import check_hardening
import pytest

pytestmark = pytest.mark.skipif(
    not (shutil.which("gcc") and shutil.which("readelf")), reason="needs gcc and readelf (binutils)"
)

# A local buffer (canary) and a copy with a known destination size (FORTIFY), so that every property has a witness
# Forty functions that stay functions (called through a table), so that endbr64 per function is a measurable ratio
FUNCTIONS = "".join(f"__attribute__((noinline)) int f{i}(int x) {{ return x + {i}; }}\n" for i in range(40))
TABLE = "int (*table[])(int) = {" + ", ".join(f"f{i}" for i in range(40)) + "};\n"
SOURCE = (
    FUNCTIONS
    + TABLE
    + """
#include <stdio.h>
#include <string.h>
int main(int argc, char** argv) {
    char buf[64];
    strcpy(buf, argv[0]);
    printf("%s\\n", buf);
    return table[argc % 40](argc) > 99;
}
"""
)

HARDENED = [
    "-O2",
    "-fPIE",
    "-pie",
    "-U_FORTIFY_SOURCE",
    "-D_FORTIFY_SOURCE=3",
    "-fstack-protector-strong",
    "-fcf-protection=full",
    "-Wl,-z,relro,-z,now,-z,noexecstack",
]
PLAIN = [
    "-O2",
    "-fno-pie",
    "-no-pie",
    "-U_FORTIFY_SOURCE",
    "-fno-stack-protector",
    "-fcf-protection=none",
    "-Wl,-z,norelro,-z,lazy,-z,execstack",
]


def build(tmp_path, flags):
    src = tmp_path / "t.c"
    src.write_text(SOURCE)
    exe = tmp_path / "t"
    subprocess.run(["gcc", *flags, str(src), "-o", str(exe)], check=True, capture_output=True)
    return exe


def test_a_hardened_program_passes(tmp_path):
    exe = build(tmp_path, HARDENED)
    missing = check_hardening.check(exe)
    # The compiler of the test machine may not know CET (non-x86 hosts): everything else must hold
    assert [m for m in missing if "CET" not in m and "BTI" not in m] == []
    if not missing:
        assert check_hardening.main([str(exe)]) == 0


def test_every_missing_property_is_named(tmp_path):
    missing = " | ".join(check_hardening.check(build(tmp_path, PLAIN)))
    for phrase in ("not a PIE", "BIND_NOW", "no stack protector", "no FORTIFY_SOURCE", "executable stack", "no CET"):
        assert phrase in missing


def test_main_fails_for_a_plain_program_and_for_a_missing_file(tmp_path, capsys):
    assert check_hardening.main([str(build(tmp_path, PLAIN))]) == 1
    assert "NOT hardened" in capsys.readouterr().err
    assert check_hardening.main([str(tmp_path / "absent")]) == 1


def test_no_arguments_is_a_usage_error(capsys):
    assert check_hardening.main([]) == 2


@pytest.mark.skipif(not shutil.which("objcopy"), reason="needs objcopy (binutils)")
def test_one_input_without_the_property_note_fails_the_program(tmp_path):
    src = tmp_path / "t.c"
    src.write_text(SOURCE)
    obj = tmp_path / "t.o"
    compile_flags = [f for f in HARDENED if not f.startswith(("-pie", "-Wl"))]
    subprocess.run(["gcc", "-c", *compile_flags, str(src), "-o", str(obj)], check=True, capture_output=True)
    if "x86_64" not in subprocess.run(["gcc", "-dumpmachine"], check=False, capture_output=True, text=True).stdout:
        pytest.skip("the property note check is x86-64 only")
    exe = tmp_path / "t"
    subprocess.run(["gcc", "-pie", str(obj), "-o", str(exe)], check=True, capture_output=True)
    assert not [m for m in check_hardening.check(exe) if "property" in m]
    subprocess.run(["objcopy", "--remove-section", ".note.gnu.property", str(obj)], check=True)
    subprocess.run(["gcc", "-pie", str(obj), "-o", str(exe)], check=True, capture_output=True)
    assert any("IBT and SHSTK" in m for m in check_hardening.check(exe))
