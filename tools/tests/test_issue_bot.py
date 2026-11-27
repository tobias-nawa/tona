"""tools/issue_bot.py: template check, reminder and close decisions."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import issue_bot as bot

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = bot.load_templates(ROOT / ".github" / "ISSUE_TEMPLATE")
NOW = datetime(2026, 12, 31, 12, 0, tzinfo=UTC)

FEATURE = (
    "### The problem\n\n{problem}\n\n### Proposed change\n\n{proposal}\n\n"
    "### Effect on speed, quality and the reference platform\n\n_No response_\n\n### AI use\n\n"
    "- [{box}] I wrote or verified this proposal myself; it is not an unchecked AI text.\n"
)


def feature_body(problem="Real problem", proposal="A change", box="x"):
    return FEATURE.format(problem=problem, proposal=proposal, box=box)


def issue(body, labels=("feature",), title="Change: something", **extra):
    base = {
        "number": 7,
        "state": "open",
        "title": title,
        "body": body,
        "labels": [{"name": n} for n in labels],
        "user": {"login": "reporter"},
        "created_at": "2026-12-01T00:00:00Z",
    }
    return base | extra


def bot_comment(text, cid=11, created="2026-12-01T00:05:00Z"):
    return {"id": cid, "body": text, "user": {"login": bot.BOT_LOGIN}, "created_at": created, "event": "commented"}


def ops(plan):
    return [a["op"] for a in plan]


def test_templates_parse_required_fields():
    feature = next(t for t in TEMPLATES if t["title"] == "Change: ")
    required = {f["label"]: f["required"] for f in feature["fields"]}
    assert required["The problem"] and required["Proposed change"]
    assert not required["Effect on speed, quality and the reference platform"]
    assert feature["labels"] == ["feature"]


def test_complete_issue_gets_nothing():
    assert bot.check_plan(issue(feature_body()), [], TEMPLATES) == []


def test_empty_field_sets_label_and_lists_it():
    plan = bot.check_plan(issue(feature_body(problem="")), [], TEMPLATES)
    assert ops(plan) == ["add_label", "comment"]
    text = plan[1]["body"]
    assert text.startswith(bot.CHECK_HEADER) and "- The problem" in text and "Proposed change\n" not in text
    assert "Label set by the bot: yes." in text


def test_no_response_counts_as_empty():
    plan = bot.check_plan(issue(feature_body(proposal="_No response_")), [], TEMPLATES)
    assert "- Proposed change" in plan[1]["body"]


def test_unticked_ai_box_is_left_to_the_author_label_and_optional_field_is_not_missing():
    assert bot.check_plan(issue(feature_body(box=" ")), [], TEMPLATES) == []
    plan = bot.check_plan(issue(feature_body(problem="", box=" ")), [], TEMPLATES)
    assert "- The problem" in plan[1]["body"]
    assert "- AI use" not in plan[1]["body"] and "Effect on speed" not in plan[1]["body"]


def test_template_found_by_label_when_prefix_was_removed():
    plan = bot.check_plan(issue(feature_body(problem=""), title="renamed"), [], TEMPLATES)
    assert ops(plan) == ["add_label", "comment"]


def test_unknown_template_or_pull_request_is_ignored():
    assert bot.check_plan(issue("", labels=(), title="free text"), [], TEMPLATES) == []
    assert bot.check_plan(issue("", pull_request={}), [], TEMPLATES) == []


def test_heading_inside_code_fence_does_not_fake_a_section():
    body = feature_body(problem="```\n### Proposed change\nfake\n```", proposal="")
    plan = bot.check_plan(issue(body), [], TEMPLATES)
    assert "- Proposed change" in plan[1]["body"] and "- The problem" not in plan[1]["body"]


def test_second_edit_updates_the_comment_instead_of_a_new_one():
    first = bot.check_plan(issue(feature_body(problem="")), [], TEMPLATES)[1]["body"]
    mine = bot_comment(first)
    later = issue(feature_body(proposal=""), labels=("feature", bot.LABEL))
    plan = bot.check_plan(later, [mine], TEMPLATES)
    assert ops(plan) == ["update_comment"] and "- Proposed change" in plan[0]["body"]
    assert "Label set by the bot: yes." in plan[0]["body"]


def test_filled_in_updates_comment_and_removes_bot_label():
    mine = bot_comment(bot.check_text(["The problem"], True))
    plan = bot.check_plan(issue(feature_body(), labels=("feature", bot.LABEL)), [mine], TEMPLATES)
    assert ops(plan) == ["update_comment", "remove_label"]
    assert "Thanks, the form is complete" in plan[0]["body"]


def test_maintainer_set_needs_info_is_untouched():
    maintainer_label = issue(feature_body(problem=""), labels=("feature", bot.LABEL))
    plan = bot.check_plan(maintainer_label, [], TEMPLATES)
    assert ops(plan) == ["comment"] and "Label set by the bot: no." in plan[0]["body"]
    mine = bot_comment(plan[0]["body"])
    done = issue(feature_body(), labels=("feature", bot.LABEL))
    assert ops(bot.check_plan(done, [mine], TEMPLATES)) == ["update_comment"]


def test_copied_header_of_a_user_is_not_trusted():
    fake = bot_comment(bot.check_text([], True))
    fake["user"] = {"login": "someone"}
    assert ops(bot.check_plan(issue(feature_body(problem="")), [fake], TEMPLATES)) == ["add_label", "comment"]


def labeled(days_ago):
    created = (NOW - timedelta(days=days_ago)).isoformat()
    return {"event": "labeled", "label": {"name": bot.LABEL}, "created_at": created}


def comment_by(login, days_ago, body="text"):
    created = (NOW - timedelta(days=days_ago)).isoformat()
    return {"event": "commented", "user": {"login": login}, "body": body, "created_at": created}


def waiting(**extra):
    return issue("", labels=("bug", bot.LABEL), **extra)


def test_young_issue_is_left_alone():
    assert bot.sweep_plan(waiting(), [labeled(13)], NOW) == []


def test_reminder_due_after_14_days():
    plan = bot.sweep_plan(waiting(), [labeled(14)], NOW)
    assert ops(plan) == ["comment"] and plan[0]["body"].startswith(bot.REMINDER_HEADER)


def test_reminder_is_sent_once():
    timeline = [labeled(20), comment_by(bot.BOT_LOGIN, 6, bot.REMINDER_HEADER + "\n\nx")]
    assert bot.sweep_plan(waiting(), timeline, NOW) == []


def test_close_due_after_28_days_as_comment_plus_close():
    timeline = [labeled(29), comment_by(bot.BOT_LOGIN, 15, bot.REMINDER_HEADER + "\n\nx")]
    assert ops(bot.sweep_plan(waiting(), timeline, NOW)) == ["comment", "close"]


def test_author_comment_resets_the_count():
    timeline = [labeled(40), comment_by("reporter", 3)]
    assert bot.sweep_plan(waiting(), timeline, NOW) == []
    old_reminder = [labeled(40), comment_by(bot.BOT_LOGIN, 30, bot.REMINDER_HEADER), comment_by("reporter", 15)]
    assert ops(bot.sweep_plan(waiting(), old_reminder, NOW)) == ["comment"]


def test_other_people_do_not_reset_the_count():
    timeline = [labeled(30), comment_by("maintainer", 2), comment_by("stranger", 1)]
    assert ops(bot.sweep_plan(waiting(), timeline, NOW))[-1] == "close"


def test_only_open_issues_with_the_label_and_no_pull_requests():
    assert bot.sweep_plan(issue("", labels=("bug",)), [labeled(40)], NOW) == []
    assert bot.sweep_plan(waiting(state="closed"), [labeled(40)], NOW) == []
    assert bot.sweep_plan(waiting(pull_request={}), [labeled(40)], NOW) == []


def test_flatten_accepts_slurped_pages_and_plain_lists():
    assert bot.flatten([[{"a": 1}], [{"a": 2}]]) == [{"a": 1}, {"a": 2}]
    assert bot.flatten([{"a": 1}]) == [{"a": 1}]


def test_apply_plan_calls_the_api_per_action(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(bot, "gh_api", lambda method, path, payload: calls.append((method, path, payload)))
    plan = [
        {"op": "add_label"},
        {"op": "remove_label"},
        {"op": "comment", "body": "head\nsecond line"},
        {"op": "update_comment", "id": 5, "body": "head\nupdated"},
        {"op": "close"},
    ]
    bot.apply_plan(plan, "o/r", 7, dry_run=False)
    assert [c[0] for c in calls] == ["POST", "DELETE", "POST", "PATCH", "PATCH"]
    assert calls[3][1] == "repos/o/r/issues/comments/5" and calls[4][2]["state_reason"] == "not_planned"
    assert "#7: comment (second line)" in capsys.readouterr().out


def test_apply_plan_dry_run_calls_nothing(monkeypatch):
    monkeypatch.setattr(bot, "gh_api", lambda *a: (_ for _ in ()).throw(AssertionError("called")))
    bot.apply_plan([{"op": "close"}], "o/r", 7, dry_run=True)


def test_main_check_and_sweep_in_dry_run(tmp_path, capsys):
    import json

    (tmp_path / "issue.json").write_text(json.dumps(issue(feature_body(problem=""))), encoding="utf-8")
    (tmp_path / "comments.json").write_text("[]", encoding="utf-8")
    (tmp_path / "timeline.json").write_text("[]", encoding="utf-8")
    common = ["--repo", "o/r", "--issue", str(tmp_path / "issue.json"), "--dry-run"]
    assert bot.main(["check", *common, "--comments", str(tmp_path / "comments.json")]) == 0
    assert "#7: add_label" in capsys.readouterr().out
    sweep = ["sweep", *common, "--timeline", str(tmp_path / "timeline.json"), "--now", "2026-12-31T12:00:00Z"]
    assert bot.main(sweep) == 0
