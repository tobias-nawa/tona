"""ClassLRU of llama.cpp #31233 (src/llama-moe-cache.cpp at the head commit), optionally with the CRF admission of
#31286: one LRU list per class, class = floor(log2(CRF)) at the last use, victim = the list head with the largest
age / 2^class, never an expert the current batch selects."""

import math
from collections import OrderedDict

from policies import Cache, Crf, class_of


class ClassLruCache(Cache):
    def __init__(self, cap: int, n_class: int, half_life: float, admit: bool = False):
        super().__init__(cap)
        self.n_class = n_class
        self.admit = admit
        self.crf = Crf(half_life)
        self.lists = [OrderedDict() for _ in range(n_class)]  # head = least recently used
        self.where: dict[int, int] = {}
        self.fresh: set[int] = set()  # preloaded and not used yet: infinite age, so they leave first

    def __contains__(self, e):
        return e in self.where

    def __len__(self):
        return len(self.where)

    def _age(self, e) -> float:
        return math.inf if e in self.fresh else self.crf.clock - self.crf.last[e]

    def victim(self, protect):
        best, best_score = None, -1.0
        for c, lst in enumerate(self.lists):
            head = next((e for e in lst if e not in protect), None)
            if head is not None and self._age(head) / 2**c > best_score:
                best, best_score = head, self._age(head) / 2**c
        return best

    def remove(self, e):
        del self.lists[self.where.pop(e)][e]
        self.fresh.discard(e)

    def _put(self, e, c):
        self.lists[c][e] = 1
        self.where[e] = c

    def insert_back(self, e):
        self._put(e, 0)
        self.lists[0].move_to_end(e, last=False)
        self.fresh.add(e)

    def demand(self, union, ntok, max_new=None):
        keep = set(union)
        misses = []
        for e in union:
            self.crf.use(e)
            if e in self.where:
                c = self.where[e]
                del self.lists[c][e]
                self.fresh.discard(e)
                self._put(e, class_of(self.crf.value[e], self.n_class))
            else:
                misses.append(e)
        for e in sorted(misses):  # llama.cpp fills in expert order
            if max_new is not None and max_new <= 0:
                break
            max_new = None if max_new is None else max_new - 1
            if len(self.where) >= self.cap and not self._make_room(e, keep):
                continue
            self._put(e, class_of(self.crf.value[e], self.n_class))
        self.crf.advance(ntok)

    def _make_room(self, e, keep) -> bool:
        v = self.victim(keep)
        if v is None or (self.admit and self.crf.value[e] <= self.crf.decayed(v)):
            return False
        self.remove(v)
        return True
