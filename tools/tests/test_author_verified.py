"""The checkbox parser of the author-verified labeler (tools/author_verified.py)."""

import author_verified as av

FORM = """### What happened

Something.

### AI use

{box}

### Other

- [ ] unrelated box
"""


def body(box: str) -> str:
    return FORM.format(box=box)


def test_ticked():
    assert av.author_state(body("- [x] I verified this report myself; it is not pasted from an AI tool.")) == "verified"


def test_ticked_capital_x():
    assert av.author_state(body("- [X] I verified this proposal myself.")) == "verified"


def test_unticked():
    assert av.author_state(body("- [ ] I verified this report myself.")) == "unverified"


def test_edited_back_and_forth():
    ticked = body("- [x] I verified this report myself.")
    unticked = ticked.replace("[x]", "[ ]")
    assert [av.author_state(t) for t in (ticked, unticked, ticked)] == ["verified", "unverified", "verified"]


def test_missing_section():
    assert av.author_state("### Hardware\n\nA GPU\n\n- [ ] a box elsewhere\n") == "none"


def test_blank_issue():
    assert av.author_state("") == "none"
    assert av.author_state("just text") == "none"


def test_section_without_box():
    assert av.author_state("### AI use\n\nnone of that\n") == "none"


def test_section_last_in_body():
    assert av.author_state("### AI use\n\n- [x] done") == "verified"


def test_box_of_another_section_is_ignored():
    assert av.author_state("### Other\n\n- [x] ticked\n\n### AI use\n\n- [ ] not\n") == "unverified"


def test_comment_starts_with_marker():
    assert av.COMMENT.splitlines()[0] == av.COMMENT_FIRST_LINE


def test_main_state_reads_the_file(tmp_path, capsys):
    body = tmp_path / "body.md"
    body.write_text("### AI use\n\n- [x] I verified this report myself.\n", encoding="utf-8")
    assert av.main(["author_verified.py", "state", str(body)]) == 0
    assert capsys.readouterr().out.strip() == "verified"


def test_main_comment_prints_the_comment(capsys):
    assert av.main(["author_verified.py", "comment"]) == 0
    assert capsys.readouterr().out.strip() == av.COMMENT.strip()


def test_main_without_arguments_prints_usage(capsys):
    assert av.main(["author_verified.py"]) == 2
    assert capsys.readouterr().err
