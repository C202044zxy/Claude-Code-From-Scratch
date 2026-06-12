# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal reimplementation of Claude Code from first principles. The thesis (see `README.md`): an agentic coding agent is mostly a simple `while` loop driving an LLM, plus a small set of tools. This repo builds that skeleton and layers systems onto it, aiming for a SWE-bench number. Keep changes minimal and in the spirit of the irreducible skeleton — don't add abstractions the loop doesn't need yet.

## Commands

```bash
uv sync                                 # install deps into .venv
cp .env.example .env                    # then add ANTHROPIC_API_KEY

uv run cc "your task"                   # run the agent against the cwd
uv run cc -C /path/to/repo "task"       # run against another repo
echo "your task" | uv run cc            # task from stdin

uv run cc --model <id> --effort <low|medium|high|xhigh|max> "task"
```

Config via env (`.env`, overrides shown as defaults): `CC_MODEL=claude-opus-4-8`, `CC_EFFORT=high`, `CC_MAX_TOKENS=16000`. CLI `--model`/`--effort` flags take precedence.

Verify changes by running `uv run cc` against a real directory, and by the checks below.

## GitHub workflow

Branch is `main`, remote `origin` is `github.com/C202044zxy/Claude-Code-From-Scratch`. Only add/commit/push when the user asks.

```bash
# Run the same checks CI runs, before committing
uv run ruff check .          # lint (auto-fix: uv run ruff check --fix .)
uv run pytest                # tests live in tests/

# Stage, commit, push
git add -A
git commit -m "message"
git push                     # pushes the current branch to origin
```

CI (`.github/workflows/ci.yml`) runs on every push to `main` and on every pull request. It does `uv sync`, an import smoke check, `ruff check .`, and `pytest`. Keep it green: run the two commands above before pushing. `pytest`/`ruff` are dev dependencies (the `dev` group in `pyproject.toml`, installed by `uv sync`); ruff config and `testpaths` also live there.

**Auto-commit + push (standing authorization):** once a change is verified (the CI-equivalent checks above pass) and staged, commit it automatically without asking, then `git push` to `origin main` right after the commit. This standing approval covers commit and push only — still do not open a PR or force-push unless the user asks.

## Architecture

The whole system feeds one loop. Read these in order:

- **`src/cc/agent.py`** — the loop, and the only place that drives the model. `Agent.run(task)` sends `messages + tools + system` to the Anthropic Messages API, executes every `tool_use` block the model returns, appends results as one `user` turn, and repeats until `stop_reason != "tool_use"` (or `max_turns`, default 60). "The model controls the loop" — we never decide the next step, we just run what the model asks. Uses `thinking={"type": "adaptive"}` and `output_config={"effort": ...}`.
- **`src/cc/tools/`** — the agent's surface area. Each tool subclasses `Tool` (`base.py`): a `name`, a `description` (the model reads this to decide *when* to call it — it's a prompt, write it carefully), an `input_schema` (JSON Schema), and `run() -> str`. `tools/__init__.py::default_tools()` is the registry; **adding a capability = appending a `Tool` here**, no loop changes needed.
- **`src/cc/prompts.py`** — the system prompt (agent identity + a few hard rules). Deliberately short.
- **`src/cc/cli.py`** — entry point (`cc` script → `cc.cli:main`). Only jobs: resolve the workdir (`-C`), read the task (args or stdin), check `ANTHROPIC_API_KEY`, `os.chdir` into the workdir, run the loop, print the final text.

## Conventions that matter

- **Assistant turns are appended verbatim** (`response.content`, not reconstructed). This preserves `thinking` and `tool_use` blocks the API requires on the next request. Don't rewrite assistant content into plain strings.
- **Tools fail by raising `ToolError`** (`tools/base.py`), not by returning error strings or letting exceptions escape. The loop catches `ToolError` (and any other exception) and feeds it back as a `tool_result` with `is_error=True` so the model can recover instead of crashing. Use it for expected failures with a message that tells the model how to recover (see `edit.py`).
- **`edit.py` invariant**: `old_string` must match exactly once unless `replace_all`. Zero matches = stale model state (reject), many matches = ambiguous (reject). This forces read-before-edit. Preserve this when touching edit logic.
- The agent operates on the **current working directory** (the CLI `os.chdir`s into the workdir). Tool paths are relative to that.
- **Keep docs in sync.** After implementing a feature or changing the structure, update the relevant docs (`README.md`, this file, the design doc) in the same change.
