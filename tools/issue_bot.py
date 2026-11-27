"""Issue bot: asks for missing form sections and reminds or closes issues that wait for an answer.

Usage:
  issue_bot.py check --repo R --issue FILE --comments FILE [--templates DIR] [--dry-run]
  issue_bot.py sweep --repo R --issue FILE --timeline FILE [--dry-run] [--now ISO8601]

The FILEs are `gh api` output of the workflows (.github/workflows/issue-check.yml, issue-reminder.yml). Issue text comes
from outsiders and is data: it is only compared and counted here, never run, and never put into a shell command; the
comments the bot writes contain template labels and fixed text, not the reporter's words.

Why not actions/stale: it is a dependency, and it cannot compare an issue with the required fields of its form.
Prior art: https://github.com/actions/stale (the 14/28-day idea); GitHub issue forms render an empty field as
`_No response_`
(https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms).
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

LABEL = "needs-info"
CHECK_HEADER = "Issue bot: template check"
REMINDER_HEADER = "Issue bot: reminder"
BOT_LOGIN = "github-actions[bot]"
REMIND_AFTER = timedelta(days=14)
CLOSE_AFTER = timedelta(days=28)
NO_RESPONSE = "_No response_"
STATE_LINE = re.compile(r"^Status: (incomplete|complete)\. Label set by the bot: (yes|no)\.$", re.MULTILINE)


def _unquote(text: str) -> str:
    return text.strip().strip("\"'")


def parse_template(text: str) -> dict:
    """The few keys the check needs, read line by line: the forms use one fixed layout, so no YAML dependency."""
    template = {"title": "", "labels": [], "fields": []}
    head, _, body = text.partition("\nbody:")
    for line in head.splitlines():
        if line.startswith("title:"):
            template["title"] = _unquote(line[len("title:") :])
        elif line.startswith("labels:"):
            template["labels"] = [_unquote(x) for x in line[len("labels:") :].strip(" []").split(",") if x.strip()]
    for block in re.split(r"^  - type: ", body, flags=re.MULTILINE)[1:]:
        kind = block.splitlines()[0].strip()
        label = re.search(r"^      label: (.+)$", block, re.MULTILINE)
        if kind == "markdown" or not label:
            continue
        options = [
            (_unquote(m.group(1)), "required: true" in m.group(2))
            for m in re.finditer(r"^        - label: (.+)\n((?:          .+\n?)*)", block, re.MULTILINE)
        ]
        required = bool(re.search(r"^      required: true", block, re.MULTILINE))
        template["fields"].append(
            {"type": kind, "label": _unquote(label.group(1)), "required": required, "options": options}
        )
    return template


def load_templates(directory: Path) -> list[dict]:
    templates = []
    for path in sorted(directory.glob("*.yaml")):
        template = parse_template(path.read_text(encoding="utf-8"))
        if template["fields"]:
            templates.append(template)
    return templates


def pick_template(title: str, labels: list[str], templates: list[dict]) -> dict | None:
    """The form an issue came from: the title prefix first, then the label the form sets (only if unambiguous)."""
    for template in templates:
        if template["title"] and title.startswith(template["title"]):
            return template
    by_label = [t for t in templates if t["labels"] and set(t["labels"]) <= set(labels)]
    return by_label[0] if len(by_label) == 1 else None


def parse_sections(body: str, labels: list[str]) -> dict[str, str]:
    """`### Label` sections of a rendered form. Only headings equal to a template label count, outside code fences."""
    sections: dict[str, list[str]] = {}
    current = None
    in_fence = False
    for line in body.replace("\r\n", "\n").split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        heading = line[4:].strip() if line.startswith("### ") else None
        if not in_fence and heading in labels and heading not in sections:
            current = heading
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def _checked(value: str, option: str) -> bool:
    return re.search(rf"^- \[[xX]\] {re.escape(option)}\s*$", value, re.MULTILINE) is not None


def missing_sections(body: str, template: dict) -> list[str]:
    labels = [f["label"] for f in template["fields"]]
    sections = parse_sections(body, labels)
    missing = []
    for field in template["fields"]:
        value = sections.get(field["label"], "")
        if field["type"] == "checkboxes":
            needed = [name for name, req in field["options"] if req]
            if any(not _checked(value, name) for name in needed):
                missing.append(field["label"])
        elif field["required"] and value in ("", NO_RESPONSE):
            missing.append(field["label"])
    return missing


def bot_comment(comments: list[dict], header: str) -> dict | None:
    """Only comments of the Actions bot count; a user cannot fake the state by copying the first line."""
    found = [
        c
        for c in comments
        if (c.get("user") or {}).get("login") == BOT_LOGIN and (c.get("body") or "").startswith(header)
    ]
    return found[-1] if found else None


def check_text(missing: list[str], bot_set: bool) -> str:
    status = "incomplete" if missing else "complete"
    state = f"Status: {status}. Label set by the bot: {'yes' if bot_set else 'no'}."
    if not missing:
        return f"{CHECK_HEADER}\n{state}\n\nThanks, the form is complete now."
    items = "\n".join(f"- {name}" for name in missing)
    return (
        f"{CHECK_HEADER}\n{state}\n\nThank you for the report. These required sections of the form are empty:\n\n"
        f"{items}\n\nPlease fill them in by editing the issue; this comment updates itself. "
        "Without an answer the issue is closed after 28 days, and you can reopen it any time with the missing details."
    )


def _filled_in_plan(mine: dict | None, state: re.Match | None, has_label: bool) -> list[dict]:
    if not mine or not state or state.group(1) == "complete":
        return []
    bot_set = state.group(2) == "yes"
    actions = [{"op": "update_comment", "id": mine["id"], "body": check_text([], bot_set)}]
    if has_label and bot_set:
        actions.append({"op": "remove_label"})
    return actions


def _missing_plan(missing: list[str], mine: dict | None, state: re.Match | None, has_label: bool) -> list[dict]:
    # A needs-info that a maintainer set stays theirs; the bot only takes over a label it set itself.
    bot_set = (state is not None and state.group(2) == "yes") if has_label else True
    body = check_text(missing, bot_set)
    actions = [] if has_label else [{"op": "add_label"}]
    if mine is None:
        actions.append({"op": "comment", "body": body})
    elif mine["body"] != body:
        actions.append({"op": "update_comment", "id": mine["id"], "body": body})
    return actions


def check_plan(issue: dict, comments: list[dict], templates: list[dict]) -> list[dict]:
    """Actions for one opened or edited issue; empty when nothing is to be done."""
    if issue.get("state") != "open" or "pull_request" in issue:
        return []
    labels = [x["name"] for x in issue.get("labels", [])]
    template = pick_template(issue.get("title") or "", labels, templates)
    if template is None:
        return []
    missing = missing_sections(issue.get("body") or "", template)
    mine = bot_comment(comments, CHECK_HEADER)
    state = STATE_LINE.search(mine["body"]) if mine else None
    if not missing:
        return _filled_in_plan(mine, state, LABEL in labels)
    return _missing_plan(missing, mine, state, LABEL in labels)


def _time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def flatten(pages: list) -> list[dict]:
    """`gh api --paginate --slurp` gives a list of pages; a single list is accepted too."""
    return [item for page in pages for item in (page if isinstance(page, list) else [page])]


def waiting_since(issue: dict, timeline: list[dict]) -> datetime:
    """The later of: needs-info was set, the author last commented. Maintainer and bot comments do not count."""
    author = (issue.get("user") or {}).get("login")
    start = _time(issue["created_at"])
    for event in timeline:
        who = (event.get("user") or event.get("actor") or {}).get("login")
        set_label = event.get("event") == "labeled" and (event.get("label") or {}).get("name") == LABEL
        if set_label or (event.get("event") == "commented" and who == author):
            start = max(start, _time(event["created_at"]))
    return start


def sweep_plan(issue: dict, timeline: list[dict], now: datetime) -> list[dict]:
    if issue.get("state") != "open" or "pull_request" in issue:
        return []
    if LABEL not in [x["name"] for x in issue.get("labels", [])]:
        return []
    start = waiting_since(issue, timeline)
    age = now - start
    if age >= CLOSE_AFTER:
        text = (
            f"{REMINDER_HEADER}\n\nClosed as not planned: the missing details did not arrive within 28 days. This is "
            "not a rejection; reopen the issue any time with the missing details and it continues."
        )
        return [{"op": "comment", "body": text}, {"op": "close"}]
    reminded = any(
        e.get("event") == "commented"
        and (e.get("user") or {}).get("login") == BOT_LOGIN
        and (e.get("body") or "").startswith(REMINDER_HEADER)
        and _time(e["created_at"]) > start
        for e in timeline
    )
    if age >= REMIND_AFTER and not reminded:
        due = (start + CLOSE_AFTER).date().isoformat()
        text = (
            f"{REMINDER_HEADER}\n\nThis issue still waits for the missing details (`{LABEL}`). Without an answer it is "
            f"closed as not planned on {due}; reopening it later with the details is always possible. A comment here "
            "restarts the count."
        )
        return [{"op": "comment", "body": text}]
    return []


def gh_api(method: str, path: str, payload: dict | None) -> None:
    cmd = ["gh", "api", "--method", method, path]
    if payload is not None:
        cmd += ["--input", "-"]
    subprocess.run(cmd, input=json.dumps(payload) if payload is not None else None, text=True, check=True)


def apply_plan(plan: list[dict], repo: str, number: int, dry_run: bool) -> None:
    base = f"repos/{repo}/issues"
    for action in plan:
        op = action["op"]
        print(f"#{number}: {op}" + (f" ({action['body'].splitlines()[1]})" if "body" in action else ""))
        if dry_run:
            continue
        if op == "add_label":
            gh_api("POST", f"{base}/{number}/labels", {"labels": [LABEL]})
        elif op == "remove_label":
            gh_api("DELETE", f"{base}/{number}/labels/{LABEL}", None)
        elif op == "comment":
            gh_api("POST", f"{base}/{number}/comments", {"body": action["body"]})
        elif op == "update_comment":
            gh_api("PATCH", f"repos/{repo}/issues/comments/{action['id']}", {"body": action["body"]})
        elif op == "close":
            gh_api("PATCH", f"{base}/{number}", {"state": "closed", "state_reason": "not_planned"})


def _load(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("check", "sweep"))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--issue", required=True)
    parser.add_argument("--comments")
    parser.add_argument("--timeline")
    parser.add_argument("--templates", default=".github/ISSUE_TEMPLATE")
    parser.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN") == "true")
    parser.add_argument("--now")
    args = parser.parse_args(argv)
    issue = _load(args.issue)
    if args.mode == "check":
        plan = check_plan(issue, flatten(_load(args.comments)), load_templates(Path(args.templates)))
    else:
        now = _time(args.now) if args.now else datetime.now(UTC)
        plan = sweep_plan(issue, flatten(_load(args.timeline)), now)
    apply_plan(plan, args.repo, issue["number"], args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
