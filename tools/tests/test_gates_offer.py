import copy
import json
import shutil
from pathlib import Path

import offer
import pytest
from conftest import ROOT

CATALOG = {
    "coder": {"kind": "model", "files": [{"path": "a", "size": 1, "sha256": "x"}]},
    "mtp": {"kind": "drafter"},
}


@pytest.fixture
def data():
    return json.loads((ROOT / "data/modeldb/offer.json").read_text())


@pytest.fixture
def small(tmp_path):
    """A repository root with one source file and the record the card points to."""
    (tmp_path / "research").mkdir()
    (tmp_path / "research/m.md").write_text("x")
    (tmp_path / "data/modeldb/models").mkdir(parents=True)
    (tmp_path / "data/modeldb/models/rec.json").write_text("{}")
    return tmp_path


def card(languages):
    verdict = {"level": "open", "basis": "none", "detail": "a", "source": "research/m.md"}
    return {"name": "C", "modeldb": "rec", "languages": {x: dict(verdict) for x in languages}}


def small_offer(**changes):
    base = {
        "hw_classes": [{"id": "pc", "measured": True}],
        "languages": [{"id": "rust"}, {"id": "go"}],
        "tasks": [{"id": "coding", "status": "available"}, {"id": "images", "status": "coming"}],
        "offers": [{"task": "coding", "hw_class": "pc", "recommended": "coder"}],
        "models": {"coder": card(["rust", "go"])},
    }
    base.update(changes)
    return base


def test_the_shipped_offer_passes_every_check_but_the_schema():
    root = Path(offer.__file__).resolve().parents[2]
    data = json.loads((root / "data/modeldb/offer.json").read_text())
    import tomllib

    catalog = tomllib.loads((root / "toolchain/models.toml").read_text())
    problems = (
        offer.check_sources(root, data)
        + offer.check_models(root, data, catalog)
        + offer.check_offers(data)
        + offer.check_verdicts(data)
    )
    assert problems == []


def test_the_shipped_offer_validates_against_its_schema():
    pytest.importorskip("jsonschema")  # the pre-commit hook installs it; the test venv is hash-pinned without it
    assert offer.check_schema(ROOT, json.loads((ROOT / "data/modeldb/offer.json").read_text())) == []


def test_schema_rejects_a_coming_task_without_note_and_an_unknown_basis(data):
    pytest.importorskip("jsonschema")
    broken = copy.deepcopy(data)
    del broken["tasks"][2]["coming_note"]
    broken["models"]["coder-iq1_m"]["rates"][0]["basis"] = "guess"
    messages = " ".join(offer.check_schema(ROOT, broken))
    assert "coming_note" in messages
    assert "guess" in messages


def test_missing_source_file_is_reported_with_its_location(small):
    data = small_offer()
    data["models"]["coder"]["languages"]["go"]["source"] = "research/gone.md"
    data["offers"][0]["source"] = "https://example.org/x"
    problems = offer.check_sources(small, data)
    assert len(problems) == 1
    assert "models/coder/languages/go/source" in problems[0]
    assert "research/gone.md" in problems[0]


def test_doc_of_a_hardware_class_is_a_source_too(small):
    data = small_offer(hw_classes=[{"id": "pc", "measured": True, "doc": "docs/none.md"}])
    assert any("doc" in p for p in offer.check_sources(small, data))


def test_model_must_be_a_model_entry_of_the_catalog(small):
    data = small_offer()
    data["offers"][0]["alternative"] = "mtp"
    data["models"]["mtp"] = card(["rust", "go"])
    problems = offer.check_models(small, data, CATALOG)
    assert any("'mtp' is no model entry" in p for p in problems)
    data["offers"][0]["recommended"] = "nope"
    problems = offer.check_models(small, data, CATALOG)
    assert any("'nope' is no model entry" in p for p in problems)
    assert any("'nope' is offered but has no card" in p for p in problems)


def test_card_needs_its_modeldb_record(small):
    data = small_offer()
    data["models"]["coder"]["modeldb"] = "missing"
    assert any("modeldb record 'missing'" in p for p in offer.check_models(small, data, CATALOG))
    assert offer.check_models(small, small_offer(), CATALOG) == []


def test_offer_rules(small):
    assert offer.check_offers(small_offer()) == []
    twice = small_offer(offers=small_offer()["offers"] * 2)
    assert any("two offers" in p for p in offer.check_offers(twice))
    same = small_offer()
    same["offers"][0]["alternative"] = "coder"
    assert any("alternative equals" in p for p in offer.check_offers(same))
    coming = small_offer()
    coming["offers"].append({"task": "images", "hw_class": "pc", "recommended": "coder"})
    assert any("is coming and must have no offer" in p for p in offer.check_offers(coming))
    none = small_offer(offers=[])
    assert any("has no offer for the measured class" in p for p in offer.check_offers(none))
    unknown = small_offer()
    unknown["offers"][0]["hw_class"] = "mac"
    assert any("unknown hardware class" in p for p in offer.check_offers(unknown))


def test_unmeasured_class_needs_no_offer():
    data = small_offer(hw_classes=[{"id": "pc", "measured": False}], offers=[])
    assert offer.check_offers(data) == []


def test_verdict_rules():
    assert offer.check_verdicts(small_offer()) == []
    missing = small_offer()
    del missing["models"]["coder"]["languages"]["go"]
    assert any("no verdict for language 'go'" in p for p in offer.check_verdicts(missing))
    extra = small_offer()
    extra["models"]["coder"]["languages"]["cobol"] = copy.deepcopy(extra["models"]["coder"]["languages"]["go"])
    assert any("'cobol' is not in the language list" in p for p in offer.check_verdicts(extra))
    guess = small_offer()
    guess["models"]["coder"]["languages"]["go"]["level"] = "strong"
    assert any("needs a basis" in p for p in offer.check_verdicts(guess))
    claim = small_offer()
    claim["models"]["coder"]["languages"]["go"]["basis"] = "measured"
    assert any("open must not claim a basis" in p for p in offer.check_verdicts(claim))
    twice = small_offer(languages=[{"id": "rust"}, {"id": "rust"}, {"id": "go"}])
    assert any("appears twice" in p for p in offer.check_verdicts(twice))


def test_check_stops_at_schema_errors(small):
    pytest.importorskip("jsonschema")
    shutil.copy(ROOT / "data/modeldb/offer.schema.json", small / "data/modeldb/offer.schema.json")
    (small / "data/modeldb/offer.json").write_text("{}")
    (small / "toolchain").mkdir()
    (small / "toolchain/models.toml").write_text("")
    problems = offer.check(small)
    assert problems and all(p.startswith("schema ") for p in problems)
