import cecar_sim as cs
import numpy as np


def logits_for(top: list[int], rows: int = 1) -> np.ndarray:
    """Router logits where the experts in `top` win in the given order."""
    z = np.zeros((rows, cs.N_EXPERT), dtype=np.float32)
    for rank, e in enumerate(top):
        z[:, e] = 10.0 - rank
    return z


def test_topk_sets_best_first():
    z = logits_for([5, 3, 9])
    assert list(cs.topk_sets(z, 3)[0]) == [5, 3, 9]


def test_features_recency_and_count():
    f = cs.Features.new()
    f.use(np.array([4]))
    f.use(np.array([7]))
    x = f.vector()
    assert x[4, 0] > x[7, 0]  # expert 4 was used longer ago: larger log recency
    assert x[4, -1] == x[7, -1] == 1.0  # both used once: uses / largest use count


def test_belady_beats_lru_on_a_cyclic_trace():
    # 5 experts cycling through 4 slots: LRU always misses, Belady keeps three of them
    passes = [logits_for([t % 5]) for t in range(100)]
    lru = cs.run_policy(passes, 4, "lru", 1)[0]
    belady = cs.run_policy(passes, 4, "belady", 1)[0]
    assert lru == 0.0
    assert belady > 0.5


def test_routing_without_a_bonus_keeps_the_original_selection():
    z = logits_for(list(range(10)))[0]
    res = np.zeros(cs.N_EXPERT, bool)
    res[:10] = True
    sel = cs.select(z, res, np.full(cs.N_EXPERT, 0.5), cs.Route(10, 10, 10, False))
    assert sorted(sel) == list(range(10))


def test_pruning_drops_non_resident_experts_below_tau():
    z = logits_for(list(range(10)))[0]
    res = np.zeros(cs.N_EXPERT, bool)
    res[:3] = True
    sel = cs.select(z, res, np.zeros(cs.N_EXPERT), cs.Route(10, 4, 10, False))
    assert sorted(sel) == [0, 1, 2, 3]  # three resident ones, plus the best-ranked non-resident one within tau
