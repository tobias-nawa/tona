import json
import tomllib

import gen_offer_data as gen
import pytest
from conftest import ROOT


def shipped():
    offer = json.loads((ROOT / "data/modeldb/offer.json").read_text())
    models = tomllib.loads((ROOT / "toolchain/models.toml").read_text())
    return offer, models


def test_literal_keeps_the_source_ascii_and_the_text_utf8():
    assert gen.lit("a") == '"a"'
    assert gen.lit('say "x"\\') == '"say \\"x\\"\\\\"'
    assert gen.lit("naïve") == '"na\\303\\257ve"'
    assert gen.lit("tab\t") == '"tab\\011"'
    gen.lit("日本").encode("ascii")


def test_numbers():
    assert gen.num(None) == "std::nullopt"
    assert gen.num(70.6) == "std::optional<double>{70.6}"
    assert gen.num(1835) == "std::optional<double>{1835.0}"


def test_output_is_a_pure_function_of_the_inputs():
    offer, models = shipped()
    assert gen.generate(offer, models) == gen.generate(offer, models)
    assert gen.generate(offer, models).isascii()


def test_size_is_the_sum_of_the_pinned_files():
    _, models = shipped()
    assert gen.sizes(models, ["coder-iq1_m"]) == {"coder-iq1_m": 29608446496 + 28800138432}


def test_a_model_missing_from_the_catalog_stops_the_generator():
    with pytest.raises(SystemExit):
        gen.sizes({}, ["nope"])


def test_every_model_card_and_task_is_in_the_output():
    offer, models = shipped()
    out = gen.generate(offer, models)
    for model_id in offer["models"]:
        assert gen.lit(model_id) in out
    for task in offer["tasks"]:
        assert gen.lit(task["id"]) in out
    assert out.count("Level::") == len(offer["models"]) * len(offer["languages"])
