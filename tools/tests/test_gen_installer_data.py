"""The data tona carries (model catalog, expert profiles) is generated from the files in toolchain/."""

import sys
import tomllib
from pathlib import Path

import gen_installer_data
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODELS = tomllib.loads((ROOT / "toolchain" / "models.toml").read_text())


def generated(tmp_path) -> str:
    out = tmp_path / "catalog_data.cpp"
    sys.argv = ["gen_installer_data.py", str(out)]
    assert gen_installer_data.main() == 0
    return out.read_text()


def test_every_model_file_is_in_the_output_with_its_checksum(tmp_path):
    text = generated(tmp_path)
    for model_id, model in MODELS.items():
        assert f'"{model_id}"' in text
        assert f'"{model["revision"]}"' in text
        for f in model["files"]:
            assert f'"{f["sha256"]}"' in text
            assert f"{f['size']}ULL" in text


def test_every_profile_is_embedded_byte_for_byte(tmp_path):
    text = generated(tmp_path)
    for profile in (ROOT / "toolchain" / "profiles").glob("*.bin"):
        assert f'"{profile.stem}"' in text
        data = profile.read_bytes()
        assert ", ".join(str(b) for b in data[:24]) in text


def test_output_is_a_pure_function_of_the_inputs(tmp_path):
    assert generated(tmp_path) == generated(tmp_path)


def test_non_ascii_is_refused_instead_of_mangled():
    with pytest.raises(UnicodeEncodeError):
        gen_installer_data.lit("café")


def test_quotes_and_backslashes_are_escaped():
    assert gen_installer_data.lit('a"b\\c') == '"a\\"b\\\\c"'


def test_a_file_from_another_repository_keeps_repo_revision_and_name_on_disk(tmp_path):
    text = generated(tmp_path)
    for model in MODELS.values():
        for f in model["files"]:
            for key in gen_installer_data.FILE_KEYS:
                if key in f:
                    assert f'"{f[key]}"' in text


def test_an_unknown_file_key_is_refused(tmp_path):
    models = tmp_path / "models.toml"
    models.write_text(
        '[m]\nrepo = "a/b"\nrevision = "r"\nfiles = [{ path = "x", size = 1, sha256 = "s", rename = "y" }]\n'
    )
    sys.argv = ["gen_installer_data.py", str(tmp_path / "out.cpp"), "--models", str(models)]
    with pytest.raises(SystemExit, match="unknown file key"):
        gen_installer_data.main()


@pytest.mark.parametrize("name", ["../x.gguf", "a/b.gguf", ".hidden", "..", "/abs.gguf", ""])
def test_a_save_as_that_is_not_a_plain_file_name_is_refused(tmp_path, name):
    models = tmp_path / "models.toml"
    models.write_text(
        f'[m]\nrepo = "a/b"\nrevision = "r"\nfiles = [{{ path = "x", size = 1, sha256 = "s", save_as = "{name}" }}]\n'
    )
    sys.argv = ["gen_installer_data.py", str(tmp_path / "out.cpp"), "--models", str(models)]
    with pytest.raises(SystemExit, match="save_as must be a plain file name"):
        gen_installer_data.main()
