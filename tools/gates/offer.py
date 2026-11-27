"""Offer gate: data/modeldb/offer.json is valid, every source exists and every model id is in the catalog.

Usage: offer.py [--root DIR]

The installer and the client show the numbers of this file ("one value, one source"), so a source that points nowhere,
a model the installer cannot fetch or a language without a verdict is a defect of the release, not of the page. Checks:

- the file validates against data/modeldb/offer.schema.json (needs the `jsonschema` package; the pre-commit hook
  installs it);
- every `source` that is no URL is a file of the repository (a path with `#` or `:` suffix is not allowed);
- every model id of `offers` and `models` is an entry of toolchain/models.toml with kind "model", and the record named
  in `modeldb` exists;
- an offer names a known task and hardware class, an available task has an offer for every measured class and a
  coming task has none, `alternative` differs from `recommended`, and one task and class have one offer;
- every model has a verdict for every language of the list; `open` carries basis `none` or `provisional`, `strong` and
  `weak` carry a basis that names where the verdict comes from.
"""

import argparse
import json
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

URL_PREFIXES = ("https://", "http://")


def sources(node, where: str = "") -> Iterator[tuple[str, str]]:
    """Every (location, value) of a `source` or `doc` key below node."""
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{where}/{key}"
            if key in ("source", "doc") and isinstance(value, str):
                yield here, value
            else:
                yield from sources(value, here)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from sources(value, f"{where}[{i}]")


def check_sources(root: Path, offer: dict) -> list[str]:
    problems = []
    for where, value in sources(offer):
        if value.startswith(URL_PREFIXES):
            continue
        if not (root / value).is_file():
            problems.append(f"{where}: source {value!r} is no file of the repository")
    return problems


def check_models(root: Path, offer: dict, catalog: dict) -> list[str]:
    problems = []
    ids = set(offer["models"])
    for task_offer in offer["offers"]:
        for key in ("recommended", "alternative"):
            if key in task_offer:
                ids.add(task_offer[key])
    for model_id in sorted(ids):
        entry = catalog.get(model_id)
        if entry is None or entry.get("kind") != "model":
            problems.append(f"model {model_id!r} is no model entry of toolchain/models.toml")
        if model_id not in offer["models"]:
            problems.append(f"model {model_id!r} is offered but has no card under `models`")
    for model_id, card in offer["models"].items():
        if not (root / "data/modeldb/models" / f"{card['modeldb']}.json").is_file():
            problems.append(f"models/{model_id}: modeldb record {card['modeldb']!r} does not exist")
    return problems


def check_offers(offer: dict) -> list[str]:
    problems = []
    tasks = {t["id"]: t for t in offer["tasks"]}
    classes = {c["id"]: c for c in offer["hw_classes"]}
    seen = set()
    for o in offer["offers"]:
        key = (o["task"], o["hw_class"])
        if key in seen:
            problems.append(f"offers: two offers for {key}")
        seen.add(key)
        if o["task"] not in tasks:
            problems.append(f"offers: unknown task {o['task']!r}")
        elif tasks[o["task"]]["status"] != "available":
            problems.append(f"offers: task {o['task']!r} is coming and must have no offer")
        if o["hw_class"] not in classes:
            problems.append(f"offers: unknown hardware class {o['hw_class']!r}")
        if o.get("alternative") == o["recommended"]:
            problems.append(f"offers {key}: the alternative equals the recommendation")
    for task_id, task in tasks.items():
        for class_id, hw in classes.items():
            if task["status"] == "available" and hw["measured"] and (task_id, class_id) not in seen:
                problems.append(f"tasks: {task_id!r} is available but has no offer for the measured class {class_id!r}")
    return problems


def check_verdicts(offer: dict) -> list[str]:
    problems = []
    languages = [lang["id"] for lang in offer["languages"]]
    if len(set(languages)) != len(languages):
        problems.append("languages: an id appears twice")
    for model_id, card in offer["models"].items():
        for lang in languages:
            verdict = card["languages"].get(lang)
            if verdict is None:
                problems.append(f"models/{model_id}: no verdict for language {lang!r} (use level open)")
                continue
            known = verdict["level"] != "open"
            if known and verdict["basis"] == "none":
                problems.append(f"models/{model_id}/{lang}: a verdict needs a basis, not none")
            if not known and verdict["basis"] not in ("none", "provisional"):
                problems.append(f"models/{model_id}/{lang}: open must not claim a basis ({verdict['basis']})")
        for lang in card["languages"]:
            if lang not in languages:
                problems.append(f"models/{model_id}: language {lang!r} is not in the language list")
    return problems


def check_schema(root: Path, offer: dict) -> list[str]:
    try:
        import jsonschema
    except ImportError:
        return ["the jsonschema package is missing (pip install jsonschema)"]
    schema = json.loads((root / "data/modeldb/offer.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    return [f"schema {'/'.join(map(str, e.absolute_path))}: {e.message}" for e in validator.iter_errors(offer)]


def check(root: Path) -> list[str]:
    offer = json.loads((root / "data/modeldb/offer.json").read_text())
    problems = check_schema(root, offer)
    if problems:  # the other checks index into the structure the schema guards
        return problems
    catalog = tomllib.loads((root / "toolchain/models.toml").read_text())
    return check_sources(root, offer) + check_models(root, offer, catalog) + check_offers(offer) + check_verdicts(offer)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    problems = check(Path(ap.parse_args().root))
    for p in problems:
        print(f"offer: {p}", file=sys.stderr)
    if problems:
        print(f"offer: {len(problems)} violation(s). See data/modeldb/README.md (offer data).", file=sys.stderr)
        return 1
    print("offer: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
