from datetime import date

import prior_art
import prior_art_links as links
import pytest

TODAY = date(2026, 11, 25)


def form(text: str) -> list[str]:
    return links.form_problems(text, TODAY)


@pytest.mark.parametrize(
    "text",
    [
        "https://arxiv.org/abs/2608.07911",
        "https://arxiv.org/abs/2611.00001v3",
        "https://arxiv.org/pdf/2501.12345v2.pdf",
        "arXiv:2410.01234",
        "https://arxiv.org/abs/1501.00001",
        "https://arxiv.org/abs/1412.6980",
        "https://arxiv.org/abs/hep-th/9901001",
        "https://arxiv.org/abs/cs.LG/0601001v2",
        "https://github.com/ggml-org/llama.cpp/pull/12345",
        "https://github.com/ggml-org/llama.cpp/issues/9",
        "https://github.com/ggml-org/llama.cpp/commit/abc1234",
        "https://github.com/ArthurSonzogni/FTXUI/blob/main/README.md#usage",
        "https://github.com/ggml-org/llama.cpp",
        "https://github.com/ggml-org",
        "see https://github.com/ggml-org/llama.cpp/pull/12345).",
    ],
)
def test_possible_links_pass(text):
    assert form(text) == []


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("https://arxiv.org/abs/2613.00001", "not a possible identifier"),
        ("https://arxiv.org/abs/2600.00001", "not a possible identifier"),
        ("https://arxiv.org/abs/2701.00001", "dated after today"),
        ("https://arxiv.org/abs/2611.99999", None),
        ("https://arxiv.org/abs/2608.0791", "wrong number of digits"),
        ("https://arxiv.org/abs/2608.00000", "wrong number of digits"),
        ("https://arxiv.org/abs/1412.69801", "wrong number of digits"),
        ("https://arxiv.org/abs/not-an-id", "not an arXiv identifier"),
        ("arXiv:12345", "not an arXiv identifier"),
        ("https://arxiv.org/abs/hep-th/9913001", "not a possible identifier"),
        ("https://github.com/a/b/pull/abc", "not a positive integer"),
        ("https://github.com/a/b/pull/0", "not a positive integer"),
        ("https://github.com/a/b/issues/12x", "not a positive integer"),
        ("https://github.com/a/b/commit/xyz", "not a hexadecimal commit hash"),
        ("https://github.com/a/b/commit/abc12", "not a hexadecimal commit hash"),
        ("https://github.com/-bad/repo", "not a possible repository name"),
        ("https://github.com/a--b/repo", "not a possible repository name"),
    ],
)
def test_impossible_links_are_reported(text, fragment):
    found = form(text)
    if fragment is None:
        assert found == []  # a possible form that may still not exist: the online check's business
    else:
        assert len(found) == 1 and fragment in found[0]


def test_prior_art_check_includes_the_form_of_links():
    text = "Change type: none\n\n## Prior art\n\nhttps://arxiv.org/abs/2613.00001\n"
    assert any("arXiv:2613.00001" in p for p in prior_art.check(text, ["docs/x.md"]))


def test_links_in_comments_do_not_count():
    text = "Change type: none\n<!-- https://arxiv.org/abs/2613.00001 -->\n"
    assert prior_art.check(text, ["docs/x.md"]) == []


def test_online_findings_and_warnings():
    text = "https://arxiv.org/abs/2608.07911 https://github.com/a/b/pull/1 https://github.com/a/b/pull/2 https://github.com/c/d"
    answers = {
        "https://arxiv.org/abs/2608.07911": 200,
        "https://github.com/a/b/pull/1": 404,
        "https://github.com/a/b/pull/2": 429,
        "https://github.com/c/d": None,
    }
    found, warnings = links.online_problems(text, fetch=answers.__getitem__, pause=0)
    assert found == ["https://github.com/a/b/pull/1 does not exist (404)"]
    assert len(warnings) == 2
    assert all("not a finding" in w for w in warnings)


def test_online_check_is_capped_and_normalises_urls():
    text = " ".join(f"https://github.com/a/b/pull/{n}" for n in range(1, 100))
    seen = []
    links.online_problems(text, fetch=lambda url: seen.append(url) or 200, pause=0)
    assert len(seen) == links.MAX_ONLINE
    assert links.links("arXiv:2501.12345v2.pdf https://github.com/a/b/blob/x/y.md#L3") == [
        "https://arxiv.org/abs/2501.12345v2",
        "https://github.com/a/b/blob/x/y.md",
    ]


def test_main_online_never_fails_on_network_errors(monkeypatch, tmp_path, capsys):
    body = tmp_path / "body.md"
    body.write_text("Change type: none\n\nSee https://github.com/a/b/pull/5\n")
    monkeypatch.setattr(links, "fetch_status", lambda url: None)
    monkeypatch.setattr(links.time, "sleep", lambda s: None)
    monkeypatch.setattr(links, "online_problems", lambda text, fetch=links.fetch_status, pause=0: ([], ["w"]))
    assert prior_art.main(["--body-file", str(body), "--files", "docs/x.md", "--online"]) == 0
    assert "warning: w" in capsys.readouterr().err
    monkeypatch.setattr(links, "online_problems", lambda text, fetch=links.fetch_status, pause=0: (["gone"], []))
    assert prior_art.main(["--body-file", str(body), "--files", "docs/x.md", "--online"]) == 1
