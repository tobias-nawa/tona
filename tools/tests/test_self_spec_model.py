"""Rates of the layer-skip draft model: hand-computed cases."""

import json

import layer_cost
import self_spec_model as m


def test_run_survival_counts_leading_window():
    samples = [[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    assert m.run_survival(samples, 0, 2) == 0.5
    assert m.run_survival(samples, 1, 2) == 0.5
    assert m.run_survival([], 0, 1) == 0.0


def test_survival_curve_is_monotone():
    curve = m.survival_curve([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]], 4)
    assert curve == sorted(curve, reverse=True)
    assert curve[0] == 2 / 3


def test_pass_ms_extends_linearly():
    assert m.pass_ms(2) == 28.9
    assert abs(m.pass_ms(10) - (52.6 + 2 * (52.6 - 49.6))) < 1e-9


def test_rate_without_drafts_is_one_token_per_pass():
    assert abs(m.rate(0, 0, [], [], 11.85) - 1 / 23.7) < 1e-12


def test_perfect_skip_draft_alone():
    # e = 2 drafts, all accepted: 3 tokens for a width-3 pass and two skip steps of half a pass
    skip = [[1, 1, 1]] * 4
    want = 3 / (32.5 + 2 * 11.85)
    assert abs(m.rate(0, 2, [], skip, 11.85) - want) < 1e-12


def test_expensive_skip_never_wins_over_cheap_mtp():
    mtp = [0.8, 0.6, 0.45, 0.3]
    skip = [[1, 1, 1, 1, 1, 1]] * 10  # even a perfect extension
    _r, k, e = m.best(mtp, skip, 11.85)
    assert e == 0 and k > 0


def test_cheap_perfect_skip_wins():
    mtp = [0.8, 0.6, 0.45, 0.3]
    skip = [[1] * 8] * 10
    _, _, e = m.best(mtp, skip, 0.24)
    assert e > 0


def test_load_drafts(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text(
        json.dumps({"kind": "seq", "ids": [1]})
        + "\n"
        + json.dumps({"kind": "draft", "mask": "a", "match": [1, 0]})
        + "\n"
    )
    assert m.load_drafts(str(p)) == {"a": [[1, 0]]}


def test_layer_costs_from_tensor_directory():
    g = {
        "kv": {"general.architecture": "x", "x.expert_used_count": 2, "x.expert_count": 4},
        "tensors": [
            {"name": "token_embd.weight", "dims": [256, 10], "type": 0, "offset": 0},
            {"name": "output.weight", "dims": [256, 10], "type": 0, "offset": 0},
            {"name": "blk.0.attn_q.weight", "dims": [256, 4], "type": 0, "offset": 0},
            {"name": "blk.0.ffn_up_exps.weight", "dims": [256, 2, 4], "type": 0, "offset": 0},
            {"name": "blk.1.ssm_out.weight", "dims": [256, 1], "type": 0, "offset": 0},
        ],
    }
    c = layer_cost.layer_costs(g)
    assert c["tail"] == 256 * 10 * 4
    assert c["layers"][0] == {"layer": 0, "dense": 256 * 4 * 4, "expert": 256 * 2 * 4, "kind": "qsa"}
    assert c["layers"][1]["kind"] == "deltanet"
    assert layer_cost.decode_bytes(c, {0}) == c["tail"] + 256 * 4 * 4 + 2 * 256 * 2 * 4


def test_prompts_follow_the_bench_set(tmp_path, monkeypatch):
    import self_spec_prompts as sp

    sets = sp.bench_prompts()
    assert len(sets["code"]) == 11 and len(sets["text"]) == 3
    monkeypatch.setattr("sys.argv", ["x", str(tmp_path)])
    assert sp.main() == 0
    first = (tmp_path / "code00.txt").read_text()
    assert first.startswith("<|im_start|>user\nWrite a Python function") and first.endswith("</think>\n\n")


def test_parse_mask_and_step_cost():
    mask = m.parse_mask("a skip=1,2 topk=4")
    assert mask["kept"] == set(range(48)) - {1, 2} and mask["n_used"] == 4
    assert m.parse_mask("b routed=off")["n_used"] == 0
    c = {
        "n_used": 10,
        "tail": 100,
        "layers": [{"layer": 0, "dense": 300, "expert": 10}, {"layer": 1, "dense": 300, "expert": 10}],
    }
    full = m.parse_mask("f", 2)
    assert abs(m.step_ms(c, full) - 23.7) < 1e-9
    one = m.parse_mask("g skip=1", 2)
    assert abs(m.step_ms(c, one) - 23.7 * (100 + 300 + 100) / 900) < 1e-9
    # cold experts: the class factor applies to the expert bytes only
    assert abs(m.step_ms(c, full, scale=2.0) - (23.7 + 23.7)) < 1e-9


def test_breakeven_step_cost():
    mtp = [0.8, 0.6, 0.45, 0.3]
    skip = [[1] * 8] * 10
    be = m.breakeven_step_ms(mtp, skip)
    assert 0 < be < 30
    assert m.best(mtp, skip, be * 0.9)[2] > 0 and m.best(mtp, skip, be * 1.1)[2] == 0
    assert m.breakeven_step_ms(mtp, []) == 0.0
