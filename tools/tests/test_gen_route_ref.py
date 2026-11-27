"""The reference file of the routing policies is what gen_route_ref.py writes; the references obey the rules."""

import contextlib
import io
import math

import gen_route_ref
from conftest import ROOT


def generated() -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        gen_route_ref.main()
    return out.getvalue()


def test_file_is_current():
    """Cases and experts exactly; weights within 4 ulp, because math.exp of macOS's libm differs from glibc's in the
    last place (1 weight off by 1 ulp on an M1 Max)."""
    got = generated().splitlines()
    want = (ROOT / "engine/tests/data/route_ref.txt").read_text().splitlines()
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        if not w.startswith("weights "):
            assert g == w
            continue
        gw, ww = [float(v) for v in g.split()[1:]], [float(v) for v in w.split()[1:]]
        assert len(gw) == len(ww)
        assert all(abs(a - b) <= 4 * math.ulp(b) for a, b in zip(gw, ww, strict=True)), (g, w)


def test_every_case_has_its_experts_and_weights():
    lines = generated().splitlines()
    assert len(lines) == 3 * len(gen_route_ref.CASES) * gen_route_ref.TOKENS
    for case, experts, weights in zip(lines[0::3], lines[1::3], lines[2::3], strict=True):
        k = int(case.split()[3])
        assert len(experts.split()) - 1 == len(weights.split()) - 1 == k


def test_softmax_weights_sum_to_one_and_sigmoid_to_scale():
    lines = generated().splitlines()
    for case, _, weights in zip(lines[0::3], lines[1::3], lines[2::3], strict=True):
        fields = case.split()
        want = 1.0 if fields[4] == "softmax" else float(fields[5])
        assert math.isclose(sum(float(w) for w in weights.split()[1:]), want, rel_tol=1e-9)


def test_ties_go_to_the_smaller_index():
    assert gen_route_ref.rank([1.0, 3.0, 3.0, 2.0, 3.0], 3) == [1, 2, 4]
    assert gen_route_ref.rank([0.5] * 8, 4) == [0, 1, 2, 3]


def test_bias_decides_the_choice_not_the_weight():
    r = [0.0, 2.0, 1.0, -1.0]
    order, weights, _ = gen_route_ref.sigmoid_case(r, [5.0, 0.0, 0.0, 0.0], 2, 1.0)
    assert order == [0, 1]
    s0, s1 = (1 / (1 + math.exp(-v)) for v in (0.0, 2.0))
    assert math.isclose(weights[0], s0 / (s0 + s1))
