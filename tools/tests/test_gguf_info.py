import io
import json
import struct
import sys

import gguf_info
import pytest


def string(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def kv(key: str, vtype: int, payload: bytes) -> bytes:
    return string(key) + struct.pack("<I", vtype) + payload


def tensor(name: str, dims: list[int], ttype: int, offset: int) -> bytes:
    return (
        string(name)
        + struct.pack("<I", len(dims))
        + b"".join(struct.pack("<Q", d) for d in dims)
        + struct.pack("<IQ", ttype, offset)
    )


@pytest.fixture
def gguf_file(tmp_path):
    long_array = struct.pack("<IQ", gguf_info.F32, 100) + struct.pack("<100f", *range(100))
    short_array = struct.pack("<IQ", gguf_info.I32, 3) + struct.pack("<3i", 1, 2, 3)
    str_array = struct.pack("<IQ", gguf_info.STR, 2) + string("a") + string("é")
    kvs = [
        kv("general.architecture", gguf_info.STR, string("test")),
        kv("test.count", gguf_info.U32, struct.pack("<I", 48)),
        kv("test.rope", gguf_info.F32, struct.pack("<f", 0.5)),
        kv("test.flag", gguf_info.BOOL, struct.pack("<?", True)),
        kv("test.short", gguf_info.ARR, short_array),
        kv("test.strings", gguf_info.ARR, str_array),
        kv("test.long", gguf_info.ARR, long_array),
    ]
    tensors = [
        tensor("a.weight", [64, 4], 8, 0),  # Q8_0: 256 elements = 8 blocks of 34 bytes
        tensor("b.weight", [256, 2], 12, 0),  # Q4_K
        tensor("c.weight", [8], 99, 0),  # a type the table does not know
    ]
    data = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + b"".join(tensors)
    path = tmp_path / "t.gguf"
    path.write_bytes(data)
    return path


def test_read_gguf(gguf_file):
    g = gguf_info.read_gguf(str(gguf_file))
    assert g["version"] == 3 and len(g["tensors"]) == 3
    kvs = g["kv"]
    assert kvs["general.architecture"] == "test" and kvs["test.count"] == 48 and kvs["test.flag"] is True
    assert kvs["test.rope"] == 0.5 and kvs["test.short"] == [1, 2, 3] and kvs["test.strings"] == ["a", "é"]
    assert kvs["test.long"] == {"array": "skipped", "type": gguf_info.F32, "len": 100}
    assert g["tensors"][0] == {"name": "a.weight", "dims": [64, 4], "type": 8, "offset": 0}


def test_wrong_magic(tmp_path):
    path = tmp_path / "x.gguf"
    path.write_bytes(b"GGML" + b"\0" * 20)
    with pytest.raises(ValueError, match="not a GGUF file"):
        gguf_info.read_gguf(str(path))


def test_unknown_value_type():
    reader = gguf_info.Reader(None)
    with pytest.raises(ValueError, match="unknown GGUF value type 77"):
        reader.value(77)


def test_truncated_array_keeps_the_head():
    items = struct.pack("<IQ", gguf_info.STR, 70) + b"".join(string(str(i)) for i in range(70))
    value = gguf_info.Reader(io.BytesIO(items)).value(gguf_info.ARR)
    assert value["array"] == "truncated" and value["len"] == 70 and value["head"] == [str(i) for i in range(8)]


def test_tensor_bytes_and_summary(gguf_file):
    g = gguf_info.read_gguf(str(gguf_file))
    assert gguf_info.tensor_bytes(g["tensors"][0]) == 8 * 34
    assert gguf_info.tensor_bytes(g["tensors"][1]) == 2 * 144
    assert gguf_info.tensor_bytes(g["tensors"][2]) is None
    s = gguf_info.summary(g)
    assert s["tensors"] == 3 and s["unknown_types"] == [99]
    assert s["by_type"]["Q8_0"] == {"count": 1, "gib": 0.0} and s["by_type"]["unknown(99)"]["count"] == 1


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["gguf_info.py", *args])
    return gguf_info.main()


def test_main_text_output(gguf_file, monkeypatch, capsys):
    assert run_main(monkeypatch, str(gguf_file), "--tensors") == 0
    out = capsys.readouterr().out
    assert out.startswith("GGUF v3, 7 keys, 3 tensors")
    assert 'general.architecture = "test"' in out and "a.weight" in out and "Q8_0" in out


def test_main_json_output(gguf_file, monkeypatch, capsys):
    assert run_main(monkeypatch, str(gguf_file), "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["version"] == 3 and data["tensors"] is None and data["summary"]["tensors"] == 3
    assert run_main(monkeypatch, str(gguf_file), "--json", "--tensors") == 0
    assert len(json.loads(capsys.readouterr().out)["tensors"]) == 3
