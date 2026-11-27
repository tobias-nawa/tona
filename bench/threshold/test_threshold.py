"""Unit tests of the pure parts (no model, no GPU, no foreign code).

Run: python3 -m unittest discover -s bench/threshold
"""

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answers
import ctxlib
import gen_a
import stats

SRC = '''\
import re
from .helpers import base_get

LIMIT = 42


class Base:
    pass


class Child(Base):
    """A child."""

    def helper_function(self, size=10, name="x"):
        """Compute something helpful for the caller of this little thing."""
        total = 0
        for i in range(size):
            total += i
        return total


def outer(value, flag=None) -> list[int]:
    return Child().helper_function(value)
'''


class StatsTest(unittest.TestCase):
    def test_mcnemar(self):
        self.assertAlmostEqual(stats.mcnemar_exact(0, 5), 0.0625)
        self.assertEqual(stats.mcnemar_exact(3, 3), 1.0)
        self.assertEqual(stats.mcnemar_exact(0, 0), 1.0)

    def test_newcombe_symmetric_without_difference(self):
        diff, lo, hi = stats.newcombe_paired(40, 5, 5, 50)
        self.assertEqual(diff, 0)
        self.assertAlmostEqual(lo, -hi)
        self.assertLess(lo, 0)

    def test_newcombe_detects_loss(self):
        diff, _lo, hi = stats.newcombe_paired(50, 30, 0, 20)
        self.assertLess(hi, 0)
        self.assertAlmostEqual(diff, -0.3)

    def test_power_formula(self):
        self.assertEqual(stats.tasks_needed(0.1, 0.03), 872)
        self.assertAlmostEqual(stats.detectable(0.1, 1000), 0.028, places=3)


class AnswersTest(unittest.TestCase):
    def test_values(self):
        self.assertTrue(answers.is_correct("default", "'abc'", '"abc"'))
        self.assertTrue(answers.is_correct("default", "10", "`10`."))
        self.assertFalse(answers.is_correct("default", "10", "The default is 10"))
        self.assertTrue(answers.is_correct("default", "(1, 2)", "(1,2)"))

    def test_names(self):
        self.assertTrue(answers.is_correct("caller", "chunk", "Chunk()".lower()))
        self.assertTrue(answers.is_correct("caller", "chunk", "`Foo.chunk`"))
        self.assertFalse(answers.is_correct("doc", "chunk", "compact"))
        self.assertTrue(answers.is_correct("base", "t.Generic[T1, T]", "t.Generic"))

    def test_body_from_whole_definition(self):
        reply = 'Here:\n```python\ndef f(x):\n    """Doc."""\n    y = x + 1\n    return y\n```\n'
        self.assertEqual(answers.extract_body(reply, "f", 8), "        y = x + 1\n        return y\n")

    def test_body_from_method_in_class_and_bare_body(self):
        reply = "```python\nclass A:\n    def f(self):\n        return 1\n```"
        self.assertEqual(answers.extract_body(reply, "f", 4), "    return 1\n")
        self.assertEqual(answers.extract_body("```\nreturn 2\n```", "f", 4), "    return 2\n")

    def test_indented_method_and_indented_body(self):
        method = "```python\n    def f(self):\n        x = 1\n        return x\n```"
        self.assertEqual(answers.extract_body(method, "f", 8), "        x = 1\n        return x\n")
        bare = "```python\n    x = 1\n    return x\n```"
        self.assertEqual(answers.extract_body(bare, "f", 4), "    x = 1\n    return x\n")

    def test_fence_inside_docstring(self):
        reply = 'Sure:\n```python\ndef f():\n    """Mentions ``` fences."""\n    return 1\n```\n'
        self.assertEqual(answers.extract_body(reply, "f", 4), "    return 1\n")

    def test_no_body(self):
        self.assertIsNone(answers.extract_body("```python\ndef f():\n    '''only doc'''\n```", "f", 4))


class ContextTest(unittest.TestCase):
    def test_hole_keeps_signature_and_docstring(self):
        f = next(x for x in ctxlib.functions_of("m.py", SRC) if x.name == "helper_function")
        holed = ctxlib.hole_text(SRC, f)
        ast.parse(holed)
        self.assertIn('"""Compute something helpful', holed)
        self.assertIn("        ...\n", holed)
        self.assertNotIn("total += i", holed)
        self.assertIn("def outer", holed)
        self.assertEqual(f.qualname, "Child.helper_function")
        self.assertTrue(f.has_doc)

    def test_cut_top_level(self):
        cut = ctxlib.cut_top_level(SRC, 5)
        ast.parse(cut)
        self.assertNotIn("def outer", cut)
        self.assertIn("class Child", cut)


class QuestionTest(unittest.TestCase):
    def test_questions(self):
        qs = {q["kind"]: q for q in gen_a.all_questions([("m.py", SRC)])}
        self.assertEqual(qs["base"]["answer"], "Base")
        self.assertEqual(qs["constant"]["answer"], "42")
        self.assertEqual(qs["returns"]["answer"], "list[int]")
        self.assertEqual(qs["doc"]["answer"], "helper_function")
        self.assertEqual(qs["caller"]["answer"], "outer")
        self.assertEqual(qs["import"]["answer"], ".helpers")
        self.assertEqual(qs["owner"]["answer"], "Child")
        defaults = [q for q in gen_a.all_questions([("m.py", SRC)]) if q["kind"] == "default"]
        self.assertEqual({q["answer"] for q in defaults}, {"10", "'x'", "None"})


if __name__ == "__main__":
    unittest.main()
