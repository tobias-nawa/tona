"""Helper distillation: dump layout, error metric, helper file layout, and that training learns a SwiGLU target."""

import struct

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

import self_spec_helper_train as t  # noqa: E402


def test_load_dump_and_layer_data(tmp_path):
    n_layer, d, n_tok = 3, 4, 5
    a = np.arange(n_tok * n_layer * 3 * d, dtype=np.float16)
    p = tmp_path / "moe-0.f16"
    a.tofile(p)
    dump = t.load_dump(str(p), n_layer, d)
    assert dump.shape == (n_tok, n_layer, 3, d)
    x, y = t.layer_data([dump], 1, "A")
    assert x.shape == (n_tok, d) and np.array_equal(x[0], dump[0, 1, 0].astype(np.float32))
    _, yb = t.layer_data([dump], 1, "B")
    assert np.allclose(yb, y + dump[:, 1, 2].astype(np.float32))


def test_load_dump_rejects_partial_token(tmp_path):
    p = tmp_path / "bad.f16"
    np.zeros(7, dtype=np.float16).tofile(p)
    with pytest.raises(ValueError):
        t.load_dump(str(p), 3, 4)


def test_rel_error():
    y = np.ones((4, 3), dtype=np.float32)
    assert t.rel_error(0 * y, y) == 1.0
    assert t.rel_error(y, y) == 0.0


def test_parse_layers():
    assert t.parse_layers("0-2,5") == [0, 1, 2, 5]


def test_helper_file_layout(tmp_path):
    h, d = 2, 3
    layer = (np.ones((h, d)), 2 * np.ones((h, d)), 3 * np.ones((d, h)))
    p = tmp_path / "h.bin"
    t.write_helper(str(p), [layer, layer], True, d)
    raw = p.read_bytes()
    assert raw[:8] == b"TONAHLP1"
    assert struct.unpack("<4I", raw[8:24]) == (2, d, h, 1)
    assert len(raw) == 24 + 2 * 3 * h * d * 4


def test_training_fits_a_swiglu_target():
    rng = np.random.default_rng(0)
    d, h = 16, 8
    g, u, dn = (rng.normal(size=s).astype(np.float32) * 0.3 for s in ((h, d), (h, d), (d, h)))
    x = rng.normal(size=(600, d)).astype(np.float32)

    def f(a):
        gg = a @ g.T
        return ((gg / (1 + np.exp(-gg))) * (a @ u.T)) @ dn.T

    y = f(x)
    *_, err = t.train_layer((x[:500], y[:500]), (x[500:], y[500:]), h, 600, "cpu")
    assert err < 0.2
