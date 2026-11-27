#!/usr/bin/env python3
"""Create and re-seal the release manifest (schema 1, docs/maintain/release.md).

create   build the unsigned manifest of a freshly built asset directory (scripts/release-build.sh)
reseal   check every file of a downloaded draft against its manifest, then set seq, release time and expiry
         (scripts/release-sign.sh; the result is what gets signed)

The manifest is read by tona with a strict parser: no other keys, no other types. Limits are checked here too, so
that a manifest which the updater would reject is never signed.
"""

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

PRODUCT = "tona"
# One manifest lists the files of every platform; each file names its platform, "any" for the files that are the same
# on all of them (the `tona` script). A client installs the files of its own platform and of "any" and ignores the rest.
PLATFORMS = ("linux-x86_64-gfx1200", "macos-arm64", "any")
# Fields that one release has once: the per-platform builds of one tag must agree on them before they are merged
HEADER_FIELDS = (
    "schema",
    "product",
    "version",
    "seq",
    "released",
    "expires",
    "commit",
    "rocm_lock_sha256",
    "severity",
    "changelog",
)
MAX_FILES = 256
MAX_TOTAL = 4 << 30
MAX_MANIFEST = 1 << 20
SEGMENT = re.compile(r"[A-Za-z0-9._+-]+")
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class ManifestError(Exception):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_name(name: str, what: str) -> None:
    if not SEGMENT.fullmatch(name) or name in (".", ".."):
        raise ManifestError(f"{what} {name!r} contains characters outside [A-Za-z0-9._+-]")


def parse_file_spec(spec: str) -> tuple[str, str, str, str]:
    parts = spec.split(":")
    if len(parts) != 4:
        raise ManifestError(f"--file needs ASSET:PATH:MODE:PLATFORM, got {spec!r}")
    asset, path, mode, platform = parts
    if platform not in PLATFORMS:
        raise ManifestError(f"platform {platform!r} must be one of {', '.join(PLATFORMS)}")
    check_name(asset, "asset name")
    segments = path.split("/")
    if len(segments) != 2 or segments[0] not in ("bin", "runtime"):
        raise ManifestError(f"path {path!r} must be bin/<name> or runtime/<name>")
    check_name(segments[1], "path segment")
    if mode not in ("0755", "0644"):
        raise ManifestError(f"mode {mode!r} must be 0755 or 0644")
    return asset, path, mode, platform


def file_entry(directory: Path, asset: str, path: str, mode: str, platform: str) -> dict:
    source = directory / asset
    if not source.is_file() or source.is_symlink():
        raise ManifestError(f"asset {asset} is missing or not a regular file")
    return {
        "asset": asset,
        "path": path,
        "platform": platform,
        "size": source.stat().st_size,
        "sha256": sha256_of(source),
        "mode": mode,
    }


def check_paths(files: list[dict]) -> None:
    """A path is unique per platform; a file for "any" takes the path on every platform."""
    for entry in files:
        if entry["platform"] not in PLATFORMS:
            raise ManifestError(f"platform {entry['platform']!r} of {entry['asset']} is unknown")
    for platform in PLATFORMS[:-1]:
        paths = [entry["path"] for entry in files if entry["platform"] in (platform, "any")]
        if len(set(paths)) != len(paths):
            raise ManifestError(f"duplicate path for {platform}")
        if "bin/tona-server" not in paths and any(entry["platform"] == platform for entry in files):
            raise ManifestError(f"{platform} has no bin/tona-server")


def validate(manifest: dict) -> None:
    files = manifest["files"]
    if not 0 < len(files) <= MAX_FILES:
        raise ManifestError(f"{len(files)} files; allowed are 1..{MAX_FILES}")
    if sum(entry["size"] for entry in files) > MAX_TOTAL:
        raise ManifestError("total size above 4 GiB")
    assets = [entry["asset"] for entry in files]
    if len(set(assets)) != len(assets):
        raise ManifestError("duplicate asset")
    check_paths(files)
    if not VERSION.fullmatch(manifest["version"]):
        raise ManifestError(f"version {manifest['version']!r} is not X.Y.Z")
    if manifest["seq"] < 1:
        raise ManifestError("seq must be a positive integer")
    if manifest["severity"] not in ("normal", "security"):
        raise ManifestError("severity must be normal or security")
    if len(json.dumps(manifest).encode()) > MAX_MANIFEST:
        raise ManifestError("manifest above 1 MiB")


def write(manifest: dict, output: Path) -> None:
    validate(manifest)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def stamp(moment: dt.datetime) -> str:
    return moment.strftime(TIME_FORMAT)


def create(args: argparse.Namespace) -> None:
    directory = Path(args.dir)
    released = dt.datetime.fromtimestamp(int(args.source_date_epoch), dt.UTC)
    entries = [file_entry(directory, *parse_file_spec(spec)) for spec in args.file]
    manifest = {
        "schema": 1,
        "product": PRODUCT,
        "version": args.version,
        "seq": args.seq,
        "released": stamp(released),
        "expires": stamp(released + dt.timedelta(days=args.expires_days)),
        "commit": args.commit,
        "rocm_lock_sha256": args.lock_sha256,
        "severity": args.severity,
        "changelog": args.changelog,
        "files": sorted(entries, key=lambda entry: (entry["platform"], entry["path"])),
    }
    write(manifest, Path(args.output))


def reseal(args: argparse.Namespace) -> None:
    directory = Path(args.dir)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        current = file_entry(directory, entry["asset"], entry["path"], entry["mode"], entry["platform"])
        if current != entry:
            raise ManifestError(f"{entry['asset']} differs from the draft manifest (size or sha256)")
    now = dt.datetime.now(dt.UTC).replace(microsecond=0)
    manifest["seq"] = args.seq
    manifest["released"] = stamp(now)
    manifest["expires"] = stamp(now + dt.timedelta(days=args.expires_days))
    if args.severity:
        manifest["severity"] = args.severity
    write(manifest, Path(args.output))


def merge(args: argparse.Namespace) -> None:
    """One manifest from the manifests of the per-platform builds of the same tag (the release workflow)."""
    parts = [json.loads(Path(name).read_text(encoding="utf-8")) for name in args.manifest]
    merged = {key: parts[0][key] for key in parts[0] if key != "files"}
    for part in parts[1:]:
        for key in HEADER_FIELDS:
            if part[key] != merged[key]:
                raise ManifestError(f"the builds disagree on {key!r}: {merged[key]!r} and {part[key]!r}")
    files = [entry for part in parts for entry in part["files"]]
    merged["files"] = sorted(files, key=lambda entry: (entry["platform"], entry["path"]))
    write(merged, Path(args.output))


def field(manifest_path: str, name: str) -> None:
    value = json.loads(Path(manifest_path).read_text(encoding="utf-8"))[name]
    print(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("create", help="unsigned manifest of an asset directory")
    make.add_argument("--dir", required=True)
    make.add_argument("--output", required=True)
    make.add_argument("--version", required=True)
    make.add_argument("--seq", type=int, required=True)
    make.add_argument("--expires-days", type=int, required=True)
    make.add_argument("--source-date-epoch", required=True)
    make.add_argument("--commit", required=True)
    make.add_argument("--lock-sha256", required=True)
    make.add_argument("--severity", default="normal")
    make.add_argument("--changelog", default="")
    make.add_argument("--file", action="append", default=[], help="ASSET:PATH:MODE:PLATFORM, repeatable")
    make.set_defaults(run=create)

    join = sub.add_parser("merge", help="one manifest from the manifests of several platform builds of one tag")
    join.add_argument("--output", required=True)
    join.add_argument("--manifest", action="append", required=True, help="repeatable, at least two")
    join.set_defaults(run=merge)

    seal = sub.add_parser("reseal", help="verify a downloaded draft and set seq and expiry")
    seal.add_argument("--dir", required=True)
    seal.add_argument("--manifest", required=True)
    seal.add_argument("--output", required=True)
    seal.add_argument("--seq", type=int, required=True)
    seal.add_argument("--expires-days", type=int, required=True)
    seal.add_argument("--severity", choices=["normal", "security"])
    seal.set_defaults(run=reseal)

    get = sub.add_parser("get", help="print one top-level field of a manifest")
    get.add_argument("--manifest", required=True)
    get.add_argument("name")
    get.set_defaults(run=lambda a: field(a.manifest, a.name))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.run(args)
    except (ManifestError, OSError, KeyError, ValueError) as error:
        print(f"release_manifest: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
