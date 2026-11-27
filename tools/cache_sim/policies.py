"""Expert cache policies of one layer, simulated on the unions of a trace.

Preloading is the one of the server (mega_prefetch.cpp): the candidates of layer l go into layer l + 1 at the back of
the replacement order (first to leave), experts that the candidates already name and that are cached are protected.
"""

import math
from collections import OrderedDict

from route_trace import n_experts


class Cache:
    """A layer's slot pool. Subclasses give the order of replacement; preloading is shared."""

    def __init__(self, cap: int):
        self.cap = cap

    def __contains__(self, e: int) -> bool:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def demand(self, union, ntok: int, max_new: int | None = None) -> None:
        """max_new: at most that many missed experts enter (None: all, as in the old simulation)."""
        raise NotImplementedError

    def victim(self, protect) -> int | None:
        raise NotImplementedError

    def remove(self, e: int) -> None:
        raise NotImplementedError

    def insert_back(self, e: int) -> None:
        raise NotImplementedError

    def prefetch(self, cand, n_pre: int) -> None:
        protect = {e for e in cand if e in self}
        for e in [e for e in cand if e not in self][:n_pre]:
            if len(self) >= self.cap:
                v = self.victim(protect)
                if v is None:
                    break
                self.remove(v)
            self.insert_back(e)
            protect.add(e)


class Crf:
    """Combined recency and frequency (LRFU): uses decayed with a half-life in tokens, kept across evictions."""

    def __init__(self, half_life: float):
        self.half_life = half_life
        self.clock = 0
        self.value = [0.0] * n_experts()
        self.last = [0] * n_experts()

    def decayed(self, e: int) -> float:
        return self.value[e] * 2.0 ** (-(self.clock - self.last[e]) / self.half_life)

    def use(self, e: int) -> None:
        self.value[e] = self.decayed(e) + 1.0
        self.last[e] = self.clock

    def advance(self, ntok: int) -> None:
        self.clock += ntok


class LruCache(Cache):
    """LRU. With `admit_half_life` a miss whose CRF is not above the decayed CRF of its victim is not cached
    (llama.cpp #31286: it is left to the CPU)."""

    def __init__(self, cap: int, admit_half_life: float | None = None):
        super().__init__(cap)
        self.order: OrderedDict[int, int] = OrderedDict()
        self.crf = Crf(admit_half_life) if admit_half_life else None

    def __contains__(self, e):
        return e in self.order

    def __len__(self):
        return len(self.order)

    def victim(self, protect):
        return next((v for v in self.order if v not in protect), None)

    def remove(self, e):
        del self.order[e]

    def insert_back(self, e):
        self.order[e] = 1
        self.order.move_to_end(e, last=False)

    def demand(self, union, ntok, max_new=None):
        keep = set(union)
        if self.crf:
            for e in union:
                self.crf.use(e)
        for e in union:
            if e in self.order:
                self.order.move_to_end(e)
        for e in union:
            if max_new is not None and max_new <= 0:
                break
            if e not in self.order and self._admit(e, keep):
                self.order[e] = 1
                max_new = None if max_new is None else max_new - 1
        if self.crf:
            self.crf.advance(ntok)

    def _admit(self, e, keep) -> bool:
        """Makes room for e; False when e stays out."""
        if len(self.order) < self.cap:
            return True
        v = self.victim(keep)
        if v is None or (self.crf and self.crf.value[e] <= self.crf.decayed(v)):
            return False
        del self.order[v]
        return True


class LrfuCache(Cache):
    """Victim = the lowest score, a hit count that decays by half every `half_life` passes (the earlier simulation)."""

    def __init__(self, cap: int, half_life: float):
        super().__init__(cap)
        self.order: OrderedDict[int, int] = OrderedDict()
        self.score = [0.0] * n_experts()
        self.decay = 0.5 ** (1 / half_life)

    def __contains__(self, e):
        return e in self.order

    def __len__(self):
        return len(self.order)

    def victim(self, protect):
        cands = [v for v in self.order if v not in protect]
        return min(cands, key=self.score.__getitem__) if cands else None

    def remove(self, e):
        del self.order[e]

    def insert_back(self, e):
        self.order[e] = 1
        self.order.move_to_end(e, last=False)

    def demand(self, union, ntok, max_new=None):
        keep = set(union)
        self.score = [s * self.decay for s in self.score]
        for e in union:
            self.score[e] += 1
            if e in self.order:
                self.order.move_to_end(e)
        for e in union:
            if max_new is not None and max_new <= 0:
                break
            if e not in self.order:
                max_new = None if max_new is None else max_new - 1
                if len(self.order) >= self.cap:
                    v = self.victim(keep)
                    if v is None:
                        continue
                    del self.order[v]
                self.order[e] = 1


def class_of(crf_value: float, n_class: int) -> int:
    return min(n_class - 1, int(math.log2(crf_value)))
