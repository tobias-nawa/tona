"""Unit tests of the pure parts (no server, no GPU).

Run: python3 -m unittest discover -s bench/needle
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import haystack
import stats

POOL = "".join(f"// ===== f{i}.cpp =====\n" + "int x = 1;\n" * 200 for i in range(40))


class HaystackTest(unittest.TestCase):
    def test_split_pool_keeps_every_character(self):
        units = haystack.split_pool(POOL)
        self.assertEqual(len(units), 40)
        self.assertEqual("".join(units), POOL)

    def test_needles_are_deterministic_and_balanced(self):
        a, b = haystack.make_needles(3), haystack.make_needles(3)
        self.assertEqual(a, b)
        self.assertNotEqual(a, haystack.make_needles(4))
        self.assertEqual(len(a), len(haystack.BINS) * haystack.NEEDLES_PER_BIN)
        self.assertEqual(len({n.name for n in a}), len(a))
        self.assertEqual(len({n.code for n in a}), len(a))
        self.assertEqual([n.frac for n in a], sorted(n.frac for n in a))
        for bin_no, (lo, hi) in enumerate(haystack.BINS):
            in_bin = [n for n in a if n.bin == bin_no]
            self.assertTrue(all(lo <= n.frac <= hi for n in in_bin))
            self.assertEqual(sum(n.mode == "single" for n in in_bin), haystack.NEEDLES_PER_BIN // 2)

    def test_document_has_every_needle_at_its_depth(self):
        needles = haystack.make_needles(1)
        doc = haystack.build_document(haystack.split_pool(POOL), 1, 60_000, needles)
        for n in needles:
            at = doc.find(n.line())
            self.assertGreater(at, -1)
            self.assertAlmostEqual(at / len(doc), n.frac, delta=0.02)

    def test_hard_document_has_decoys_and_the_current_question(self):
        needles = haystack.make_needles(1, hard=True)
        self.assertEqual([n.code for n in needles], [n.code for n in haystack.make_needles(1)])
        doc = haystack.build_document(haystack.split_pool(POOL), 1, 60_000, needles)
        for n in needles:
            self.assertIn(n.line(), doc)
            self.assertEqual(len(n.decoys), 3)
            self.assertTrue(all(line in doc for _, line in n.decoys))
        self.assertIn("highest revision", haystack.single_question(needles[0]))
        self.assertIn("current", haystack.multi_question(needles, 1))

    def test_texts_differ(self):
        units = haystack.split_pool(POOL)
        self.assertNotEqual(
            haystack.build_document(units, 1, 30_000, []), haystack.build_document(units, 2, 30_000, [])
        )

    def test_parse_multi(self):
        got = haystack.parse_multi("amber: 123456\n- **basalt**: `654321`\nCobalt = 111111\nnone: 12345\n")
        self.assertEqual(got, {"amber": "123456", "basalt": "654321", "cobalt": "111111"})

    def test_multi_question_names_all(self):
        needles = haystack.make_needles(2)
        q = haystack.multi_question(needles, 2)
        self.assertTrue(all(n.name in q for n in needles))


class StatsTest(unittest.TestCase):
    def test_mcnemar(self):
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(stats.mcnemar_exact(0, 5), 0.0625)
        self.assertAlmostEqual(stats.mcnemar_exact(5, 5), 1.0)

    def test_paired_diff(self):
        d, lo, hi = stats.paired_diff(2, 12, 100)
        self.assertAlmostEqual(d, 0.10)
        self.assertLess(lo, d)
        self.assertGreater(lo, 0)
        self.assertGreater(hi, d)

    def test_pairs_needed_matches_a_known_value(self):
        # Connor (1987): psi 0.25, d 0.15, alpha 0.05, power 0.8 gives about 85 pairs
        self.assertIn(stats.pairs_needed(0.25, 0.15), range(83, 88))
        self.assertAlmostEqual(stats.detectable(0.25, stats.pairs_needed(0.25, 0.15)), 0.15, delta=0.005)

    def test_wilson(self):
        lo, hi = stats.wilson(8, 10)
        self.assertLess(lo, 0.8)
        self.assertGreater(hi, 0.8)
        self.assertEqual(stats.wilson(0, 0), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
