import struct

import classlru
import cli
import pytest
import route_trace as trace
import sim
from policies import Crf, LrfuCache, LruCache, class_of


def row(top: list[int], value: float = 1.0) -> bytes:
    """256 float32 logits: `top` get descending values, the rest 0."""
    vals = [0.0] * trace.N_EXPERT
    for rank, e in enumerate(top):
        vals[e] = value + len(top) - rank
    return struct.pack("<256f", *vals)


def write_trace(path, records):
    """records: (layer, [(router top list, pred top list) per token])."""
    with open(path, "wb") as f:
        for layer, tokens in records:
            f.write(bytes([layer, len(tokens)]))
            f.write(b"".join(row(r) for r, _ in tokens))
            f.write(b"".join(row(p) for _, p in tokens))


def test_layer_pass_union_and_candidates():
    lp = trace.layer_pass(3, [], [])
    assert lp.union == () and lp.cand == ()
    from array import array

    router = [
        array("f", struct.unpack("<256f", row(list(range(10))))),
        array("f", struct.unpack("<256f", row(list(range(5, 15))))),
    ]
    pred = [array("f", struct.unpack("<256f", row([100, 101, *list(range(14))])))] * 2
    lp = trace.layer_pass(3, router, pred)
    assert lp.layer == 3 and lp.ntok == 2
    assert set(lp.union) == set(range(15)) and len(lp.union) == 15
    assert lp.cand[:2] == (100, 101)
    assert len(lp.cand) == 16


def test_load_passes_groups_by_layer_zero(tmp_path):
    p = tmp_path / "t.bin"
    tok = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12])
    write_trace(p, [(0, [tok]), (1, [tok]), (0, [tok, tok]), (1, [tok])])
    passes = trace.load_passes(str(p))
    assert [len(ps) for ps in passes] == [2, 2]
    assert passes[1][0].ntok == 2


def test_truncated_record_is_an_error(tmp_path):
    p = tmp_path / "t.bin"
    p.write_bytes(bytes([0, 1]) + b"\0" * 100)
    with pytest.raises(ValueError, match="truncated"):
        trace.load_passes(str(p))


def test_lru_evicts_oldest_and_keeps_the_union():
    c = LruCache(3)
    c.demand([1, 2, 3], 1)
    c.demand([3, 4], 1)
    assert 1 not in c and 2 in c and 3 in c and 4 in c
    c.demand([5, 6, 7, 8], 1)  # more than the slots: the union cannot all stay
    assert len(c) == 3


def test_prefetch_goes_to_the_back_and_protects_cached_candidates():
    c = LruCache(3)
    c.demand([1, 2, 3], 1)
    c.prefetch([2, 9, 8], 2)  # 2 is cached and protected; 9 replaces 1, 8 replaces 3
    assert 2 in c and 9 in c and 8 in c and 1 not in c and 3 not in c
    c.demand([4], 1)  # a preloaded expert leaves first
    assert 4 in c and 2 in c and (9 not in c or 8 not in c)


def test_lru_admission_bypasses_cold_miss():
    c = LruCache(2, admit_half_life=1000)
    for _ in range(5):
        c.demand([1, 2], 1)
    c.demand([3], 1)  # crf(3) = 1, below the victim's
    assert 3 not in c and 1 in c and 2 in c


def test_crf_decays_by_half_in_a_half_life():
    crf = Crf(10)
    crf.use(0)
    crf.advance(10)
    assert crf.decayed(0) == pytest.approx(0.5)
    crf.use(0)
    assert crf.value[0] == pytest.approx(1.5)


def test_class_of_is_floor_log2_capped():
    assert [class_of(v, 4) for v in (1.0, 1.9, 2.0, 7.9, 8.0, 100.0)] == [0, 0, 1, 2, 3, 3]


def test_class_lru_protects_frequent_over_recent():
    c = classlru.ClassLruCache(2, 4, 1000)
    for _ in range(8):
        c.demand([1], 1)  # class 3
    c.demand([2], 1)
    c.demand([2], 1)  # class 1
    c.demand([3], 1)  # victim: 1 has age 3 / 2^3, 2 has age 1 / 2^1, so 2 goes though it is more recent
    assert 1 in c and 2 not in c and 3 in c


def test_class_lru_preloaded_leave_first_and_selected_are_safe():
    c = classlru.ClassLruCache(2, 4, 100)
    c.demand([1], 1)
    c.prefetch([7], 1)
    c.demand([2], 1)
    assert 7 not in c and 1 in c and 2 in c
    c.prefetch([8], 1)  # evicts the oldest not protected
    assert len(c) == 2


def test_class_lru_admission_leaves_cold_miss_out():
    c = classlru.ClassLruCache(2, 4, 1000, admit=True)
    for _ in range(4):
        c.demand([1, 2], 1)
    c.demand([3], 1)
    assert 3 not in c and 1 in c and 2 in c
    plain = classlru.ClassLruCache(2, 4, 1000)
    for _ in range(4):
        plain.demand([1, 2], 1)
    plain.demand([3], 1)
    assert 3 in plain


def test_lrfu_evicts_lowest_score():
    c = LrfuCache(2, 1000)
    c.demand([1, 2], 1)
    c.demand([1], 1)
    c.demand([3], 1)
    assert 1 in c and 2 not in c and 3 in c


def test_make_factory_and_unknown_spec():
    assert isinstance(sim.make_factory("lru", 4)(), LruCache)
    assert isinstance(sim.make_factory("lrfu:3", 4)(), LrfuCache)
    assert sim.make_factory("class-admit:8:64", 4)().admit is True
    assert sim.make_factory("lru-admit:32", 4)().crf is not None
    with pytest.raises(ValueError):
        sim.make_factory("belady", 4)


def test_hit_rate_counts_unions_after_warm_up(tmp_path):
    p = tmp_path / "t.bin"
    tok = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [20, 21])
    write_trace(p, [(0, [tok])] * 10)
    passes = trace.load_passes(str(p))
    assert len(passes) == 10
    assert sim.hit_rate(passes, sim.make_factory("lru", 16)) == 1.0
    assert sim.hit_rate(passes, sim.make_factory("lru", 4)) < 1.0


@pytest.mark.parametrize("spec", ["lru", "lrfu:10", "class:4:128"])
def test_server_fill_lets_misses_in_only_while_slots_are_free(spec):
    make = sim.make_factory(spec, 20)
    cache = make()
    cache.demand(list(range(10)), 1, 4)  # 4 per pass while slots are free (FILL_CAP)
    assert len(cache) == 4
    full = make()
    full.demand(list(range(10)), 1)  # first simulation: every miss enters
    assert len(full) == 10
    small = sim.make_factory(spec, 5)()
    for _ in range(3):
        small.demand(list(range(10)), 1, min(sim.FILL_CAP, max(small.cap - len(small), 0)))
    assert len(small) == 5  # full: nothing more enters


def test_hit_rate_server_fill_is_not_above_every_miss_in(tmp_path):
    p = tmp_path / "t.bin"
    tok = (list(range(10)), [])
    write_trace(p, [(0, [tok])] * 10)
    passes = trace.load_passes(str(p))
    every = sim.hit_rate(passes, sim.make_factory("lru", 16))
    server = sim.hit_rate(passes, sim.make_factory("lru", 16), server_fill=True)
    assert every == 1.0 and 0.0 < server <= every


def test_cli_grid_runs_and_reads_crosswise(tmp_path, capsys):
    paths = []
    for name in ("a", "b"):
        p = tmp_path / f"{name}.bin"
        tok = (list(range(10)), [20, 21])
        write_trace(p, [(0, [tok]), (1, [tok])] * 6)
        paths.append(str(p))
    assert cli.main(["--slots", "12", "--jobs", "1", "--grid", *paths]) == 0
    out = capsys.readouterr().out
    assert "tuned on" in out and "class-admit:8:512" in out


def test_crosswise_picks_best_on_tuning_trace():
    res = {("a", "x"): 1.0, ("a", "y"): 2.0, ("b", "x"): 5.0, ("b", "y"): 3.0}
    lines = cli.crosswise(res, ["a", "b"], ["x", "y"])
    assert lines[0] == "tuned on a: y 2.00 -> held out b: 3.00"
    assert lines[1] == "tuned on b: x 5.00 -> held out a: 1.00"
