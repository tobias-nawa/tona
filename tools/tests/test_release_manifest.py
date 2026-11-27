"""scripts/release_manifest.py: the manifest it writes is what tona's strict parser reads (schema 1)."""

import json
import os
from pathlib import Path

import pytest
from conftest import ROOT, load_script

rm = load_script("scripts/release_manifest.py")
GOLDEN = ROOT / "engine/tests/data/update/created-manifest.json"
LINUX, MACOS = "linux-x86_64-gfx1200", "macos-arm64"
ASSETS = {
    "tona-server": (b"server binary\n", "bin/tona-server", "0755", LINUX),
    "tona": (b"client binary\n", "bin/tona", "0755", LINUX),
    "rocm-888888888888-libhip.so.7": (b"runtime library\n", "runtime/libhip.so.7", "0644", LINUX),
    "macos-arm64-tona-server": (b"mac server binary\n", "bin/tona-server", "0755", MACOS),
    "macos-arm64-tona": (b"mac client binary\n", "bin/tona", "0755", MACOS),
    "macos-arm64-tona.metallib": (b"kernels\n", "bin/tona.metallib", "0644", MACOS),
}
# Fixed inputs: the golden file must not change from run to run
EPOCH = "1793836800"  # 2026-11-05T00:00:00Z
COMMIT = "c" * 40
LOCK = "8" * 64


def make_assets(directory: Path) -> list[str]:
    specs = []
    for asset, (content, path, mode, platform) in ASSETS.items():
        (directory / asset).write_bytes(content)
        specs += ["--file", f"{asset}:{path}:{mode}:{platform}"]
    return specs


def create_args(directory: Path, output: Path, specs: list[str], **override) -> list[str]:
    values = {"version": "0.3.0", "seq": "3", "expires-days": "180", "severity": "normal", "changelog": "Test release."}
    values.update(override)
    args = ["create", "--dir", str(directory), "--output", str(output), "--source-date-epoch", EPOCH]
    args += ["--commit", COMMIT, "--lock-sha256", LOCK]
    for key, value in values.items():
        args += [f"--{key}", value]
    return args + specs


def run(argv: list[str]) -> None:
    args = rm.build_parser().parse_args(argv)
    args.run(args)


@pytest.fixture
def created(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    specs = make_assets(assets)
    out = tmp_path / "manifest.json"
    run(create_args(assets, out, specs))
    return assets, out, specs


def test_create_writes_schema_one(created):
    _, out, _ = created
    manifest = json.loads(out.read_text())
    assert list(manifest) == [
        "schema", "product", "version", "seq", "released", "expires", "commit",
        "rocm_lock_sha256", "severity", "changelog", "files",
    ]  # fmt: skip
    assert manifest["schema"] == 1 and manifest["product"] == "tona"
    assert manifest["released"] == "2026-11-05T00:00:00Z" and manifest["expires"] == "2027-05-04T00:00:00Z"
    order = [(f["platform"], f["path"]) for f in manifest["files"]]
    assert order == sorted(order)
    for entry in manifest["files"]:
        content = ASSETS[entry["asset"]][0]
        assert entry["size"] == len(content) and entry["sha256"] == rm.sha256_of(out.parent / "assets" / entry["asset"])
        assert set(entry) == {"asset", "path", "platform", "size", "sha256", "mode"}


def test_matches_the_file_the_cpp_parser_reads(created):
    # engine/tests/test_update_manifest.cpp parses this very file; if the writer changes, regenerate it
    assert created[1].read_text() == GOLDEN.read_text()


@pytest.mark.parametrize(
    "spec",
    [
        "a:bin/x:0755",  # missing platform
        "a:bin/x:0755:linux-x86_64-gfx1200:extra",
        "a:bin/x:0755:windows",  # unknown platform
        "a:bin/sub/x:0755:any",  # three segments
        "a:x:0755:any",  # one segment
        "a:etc/x:0755:any",  # unknown area
        "a:bin/..:0755:any",
        "a:bin/.:0755:any",
        "a:/bin/x:0755:any",
        "a:bin/x y:0755:any",
        "a b:bin/x:0755:any",
        "a:bin/x:4755:any",
        "a:bin/x:0777:any",
    ],
)
def test_bad_file_specs(spec):
    with pytest.raises(rm.ManifestError):
        rm.parse_file_spec(spec)


def test_missing_and_symlinked_assets(tmp_path):
    (tmp_path / "real").write_bytes(b"x")
    os.symlink(tmp_path / "real", tmp_path / "link")
    with pytest.raises(rm.ManifestError):
        rm.file_entry(tmp_path, "nothing", "bin/x", "0755", "any")
    with pytest.raises(rm.ManifestError):
        rm.file_entry(tmp_path, "link", "bin/x", "0755", "any")
    assert rm.file_entry(tmp_path, "real", "bin/x", "0755", "any")["size"] == 1


@pytest.mark.parametrize(
    "override",
    [{"version": "1.2"}, {"version": "1.2.3-rc1"}, {"version": "../1.0.0"}, {"seq": "0"}, {"severity": "critical"}],
)
def test_create_refuses_bad_fields(tmp_path, override):
    assets = tmp_path / "assets"
    assets.mkdir()
    specs = make_assets(assets)
    out = tmp_path / "m.json"
    with pytest.raises((rm.ManifestError, SystemExit)):
        run(create_args(assets, out, specs, **override))
    assert not out.exists()


def valid_manifest(created):
    return json.loads(created[1].read_text())


def many_files(manifest: dict) -> list[dict]:
    return [dict(manifest["files"][0], path=f"bin/{i}", asset=f"a{i}") for i in range(257)]


def test_validate_limits(created):
    base = valid_manifest(created)
    rm.validate(base)
    cases = {
        "no files": lambda m: m.update(files=[]),
        "too many files": lambda m: m.update(files=many_files(m)),
        "duplicate path": lambda m: m["files"][1].update(path=m["files"][0]["path"]),
        "any and platform share a path": lambda m: m["files"][3].update(platform="any"),
        "unknown platform": lambda m: m["files"][0].update(platform="windows"),
        "platform without a server": lambda m: m["files"][4].update(path="bin/other"),
        "duplicate asset": lambda m: m["files"][1].update(asset=m["files"][0]["asset"]),
        "over 4 GiB": lambda m: m["files"][0].update(size=5 << 30),
        "over 1 MiB": lambda m: m.update(changelog="x" * (1 << 20)),
    }
    for name, edit in cases.items():
        broken = json.loads(json.dumps(base))
        edit(broken)
        with pytest.raises(rm.ManifestError):
            rm.validate(broken)
        assert name


def reseal(assets: Path, draft: Path, out: Path, seq: int, *extra: str) -> None:
    flags = ["--dir", str(assets), "--manifest", str(draft), "--output", str(out), "--seq", str(seq)]
    run(["reseal", *flags, "--expires-days", "30", *extra])


def test_reseal_checks_every_file(created, tmp_path):
    assets, draft, _ = created
    out = tmp_path / "sealed.json"
    reseal(assets, draft, out, 7, "--severity", "security")
    sealed = json.loads(out.read_text())
    assert sealed["seq"] == 7 and sealed["severity"] == "security"
    assert sealed["files"] == valid_manifest(created)["files"]
    assert sealed["released"] < sealed["expires"]
    (assets / "tona").write_bytes(b"swapped after the build\n")
    with pytest.raises(rm.ManifestError, match="tona"):
        reseal(assets, draft, out, 8)
    (assets / "tona").unlink()
    with pytest.raises(rm.ManifestError):
        reseal(assets, draft, out, 8)


def test_get_and_main(created, capsys, monkeypatch):
    _, out, _ = created
    monkeypatch.setattr("sys.argv", ["release_manifest.py", "get", "--manifest", str(out), "version"])
    assert rm.main() == 0
    assert capsys.readouterr().out.strip() == "0.3.0"
    monkeypatch.setattr("sys.argv", ["release_manifest.py", "get", "--manifest", str(out), "nothing"])
    assert rm.main() == 1
    assert "release_manifest:" in capsys.readouterr().err


def part(tmp_path: Path, name: str, platform: str, **override) -> Path:
    """The manifest one platform's build writes: only its own assets, the same header."""
    assets = tmp_path / name
    assets.mkdir()
    specs = []
    for asset, (content, path, mode, plat) in ASSETS.items():
        if plat == platform:
            (assets / asset).write_bytes(content)
            specs += ["--file", f"{asset}:{path}:{mode}:{plat}"]
    out = tmp_path / f"{name}.json"
    run(create_args(assets, out, specs, **override))
    return out


def test_merge_joins_the_platform_builds(tmp_path):
    linux, mac = part(tmp_path, "linux", LINUX), part(tmp_path, "mac", MACOS)
    merged = tmp_path / "merged.json"
    run(["merge", "--output", str(merged), "--manifest", str(linux), "--manifest", str(mac)])
    manifest = json.loads(merged.read_text())
    assert {f["platform"] for f in manifest["files"]} == {LINUX, MACOS}
    assert len(manifest["files"]) == len(ASSETS)
    rm.validate(manifest)


def test_merge_refuses_builds_that_disagree(tmp_path):
    linux = part(tmp_path, "linux", LINUX)
    mac = part(tmp_path, "mac", MACOS, version="0.3.1")
    with pytest.raises(rm.ManifestError, match="version"):
        run(["merge", "--output", str(tmp_path / "m.json"), "--manifest", str(linux), "--manifest", str(mac)])
    assert not (tmp_path / "m.json").exists()
