import json
import sys
import urllib.request

url = sys.argv[-1]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout_s": {"type": "integer"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the repo",
            "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
        },
    },
]
PROMPTS = [
    "Show me the contents of src/main.cpp.",
    "Run the unit tests with a 60 second timeout.",
    "Find where parse_header is defined.",
    "Read README.md and then tell me what it says.",
    "List the files in the current directory.",
    "Search for TODO comments.",
    "Check the git status.",
    "Open tools/build.py.",
    "What does the function load_model in engine/model/loader.cpp do? Look at the file.",
    "Run make clean and then make all.",
    "Where is the constant MAX_TOKENS used?",
    "Print the last 20 lines of build.log.",
    "Show the diff of the last commit.",
    "Find all files that import numpy.",
    "Read config/settings.toml.",
    "Run pytest -x tests/test_api.py with a 120 second timeout.",
    "Search the code for the string 'deprecated'.",
    "Show me how the CLI arguments are parsed, the entry point is cli/main.py.",
    "Count the lines of code in the src directory.",
    "Read the file Cargo.toml and tell me the dependencies.",
    "Execute npm test.",
    "Find the definition of class Scheduler.",
    "Open docs/architecture.md.",
    "Which python version does this project use? Check pyproject.toml.",
    "Run the linter on the whole project.",
    "Search for functions named handle_request.",
    "Look at the Dockerfile.",
    "Show disk usage of the build directory.",
    "Read tests/conftest.py.",
    "Find usages of the function compute_hash.",
    "Run ls -la in the home directory of the project.",
    "Check which branch we are on.",
]
ok = 0
for p in PROMPTS:
    body = json.dumps(
        {
            "model": "x",
            "messages": [{"role": "user", "content": p}],
            "tools": TOOLS,
            "max_tokens": 200,
            "temperature": 0,
        }
    ).encode()
    try:
        r = urllib.request.urlopen(
            urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"}),
            timeout=600,
        )
        m = json.load(r)["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        good = bool(calls) and all(
            isinstance(json.loads(c["function"]["arguments"]), dict)
            and c["function"]["name"] in ("read_file", "run_command", "search")
            for c in calls
        )
    except Exception:
        good = False
    ok += good
    print(int(good), p, flush=True)
print("valid tool calls", ok, "of", len(PROMPTS))
