# System Design

This document describes the architecture of **Claude Code, From Scratch** — a
minimal reimplementation of an agentic coding agent. The guiding thesis (see
`README.md`) is that such an agent is mostly a simple `while` loop driving an
LLM, surrounded by a small set of tools. Everything below feeds that loop.

## Technology Stack

| Concern | Choice | Notes |
| --- | --- | --- |
| Language | Python `>=3.11` | Uses `from __future__ import annotations` throughout. |
| LLM SDK | [`anthropic`](https://pypi.org/project/anthropic/) `>=0.40.0` | The single runtime dependency. Speaks the Anthropic Messages API. |
| Default model | `claude-opus-4-8` | Overridable via `--model` / `CC_MODEL`. |
| Packaging / env | [`uv`](https://docs.astral.sh/uv/) + `hatchling` | `uv sync` installs into `.venv`; wheel packages `src/cc`. |
| Lint | `ruff` (line length 88) | Dev dependency; run by CI. |
| Test | `pytest` (`testpaths = ["tests"]`) | Dev dependency; run by CI. |
| CI | GitHub Actions | `uv sync` → import smoke check → `ruff check .` → `pytest`, on every push to `main` and every PR. |
| Config | Env vars + tiny `.env` loader | No config framework; `.env` parsed by hand in `cli.py`. |

There is intentionally **no web framework, no database, no async runtime, and no
agent framework**. The whole program is a CLI process that makes blocking calls
to one HTTP API.

### Providers

Although the agent targets Anthropic, the loop is provider-pluggable through a
small preset table (`PROVIDERS` in `agent.py`). Both supported providers speak
the *same* Messages API — DeepSeek ships an Anthropic-compatible endpoint — so
the SDK, message format, and tool format are identical across them. Only three
things vary per provider: `base_url`, the API-key env var, and whether
Anthropic-only request params (`thinking`, `output_config`) are sent.

| Provider | Base URL | Key env | Default model | Extended params |
| --- | --- | --- | --- | --- |
| `anthropic` | SDK default | `ANTHROPIC_API_KEY` | `claude-opus-4-8` | yes |
| `deepseek` | `api.deepseek.com/anthropic` | `DEEPSEEK_API_KEY` | `deepseek-chat` | no |

Adding another compatible provider is one row in the table.

## Structure

```
src/cc/
  agent.py        # THE LOOP — the only place that drives the model
  prompts.py      # system prompt (identity + a few hard rules)
  cli.py          # entry point: resolve workdir, read task, run loop, print
  swebench.py     # SWE-bench runner: instance → agent → git diff prediction
  tools/          # the agent's surface area on the world
    base.py       #   Tool ABC + ToolError
    bash.py       #   run shell commands (the workhorse)
    read.py       #   read a file with line numbers
    write.py      #   create / overwrite a file
    edit.py       #   exact-string replacement (read-before-edit invariant)
    glob.py       #   find files by name pattern
    grep.py       #   search file contents by regex
    __init__.py   #   registry: default_tools()
tests/
  test_smoke.py   # import + basic behavior checks
```

The dependency direction is one-way and shallow: `{cli,swebench}.py → agent.py →
{prompts, tools}`. Tools never import the agent; the agent never hard-codes a
tool; both entry points drive the same unchanged loop.

## Design

### 1. The loop is the system

`Agent.run(task)` (`src/cc/agent.py`) is the heart. It implements the standard
Anthropic tool-use pattern:

```
messages = [user: task]
loop (up to max_turns, default 60):
    response = model(system, tools, messages)     # one API call
    append assistant turn verbatim
    if stop_reason != "tool_use":
        return final text                          # done
    for each tool_use block:
        result = execute(tool, input)
        collect tool_result
    append all results as ONE user turn
```

The defining principle is **"the model controls the loop."** The code never
decides the next action — it executes whatever `tool_use` blocks the model
emits and hands the results back. This keeps the orchestration logic tiny
(~50 lines) and pushes all the intelligence into the model + prompt + tool
descriptions.

Anthropic-only knobs are sent only when the provider supports them:
`thinking={"type": "adaptive"}` and `output_config={"effort": ...}`.

### 2. Conversation state is append-only and verbatim

`self.messages` is a plain list that only grows. The critical invariant:
**assistant turns are appended exactly as the API returned them**
(`response.content`), never reconstructed into plain strings. This preserves the
`thinking` and `tool_use` content blocks that the Messages API requires to be
present on the next request. All tool results from one assistant turn are
batched into a single `user` turn, matching the API's expectation.

### 3. Tools are the surface area

Each tool subclasses `Tool` (`tools/base.py`) and supplies four things:

- `name` — what the model calls.
- `description` — *a prompt*: the model reads it to decide **when** to call the
  tool. These are written carefully, not as afterthoughts.
- `input_schema` — JSON Schema for the arguments.
- `run(**kwargs) -> str` — executes and returns a string result.

`default_tools()` (`tools/__init__.py`) is the registry. **Adding a capability
is appending a `Tool` here — no loop changes.** The six default tools (a shell
plus file read/write/edit and name/content search) are the "irreducible set"
that gives enough leverage to solve real coding tasks. `bash` alone is nearly
universal; the dedicated file tools exist because they are safer and more
legible than shelling out for the same actions.

### 4. Errors are recoverable, not fatal

Tools signal expected failures by raising `ToolError` with a message that tells
the model **how to recover** — not by returning error strings or letting
exceptions escape. The loop catches `ToolError` (and any other exception) in
`_execute` and feeds it back as a `tool_result` with `is_error=True`. The model
then sees the failure and can adapt, rather than the process crashing. This
turns the LLM into the error-handling layer.

The `edit` tool illustrates the philosophy: `old_string` must match **exactly
once** (unless `replace_all`). Zero matches means the model's view of the file
is stale → reject. Many matches means the edit is ambiguous → reject. Both
rejections carry guidance, and together they *force read-before-edit*.

### 5. The CLI is a thin shell

`cli.py` does only the boring, deterministic work around the loop:

1. Load `.env` (a dependency-free parser; real env vars win).
2. Read the task from CLI args or stdin.
3. Resolve the working directory (`-C/--workdir`) and validate the API key.
4. `os.chdir` into the workdir — so **all tool paths are relative to the repo
   under work**, not the launch directory.
5. Construct the `Agent`, run it, print progress and the final answer.

Config precedence is: CLI flag → env var (`CC_MODEL`, `CC_EFFORT`,
`CC_MAX_TOKENS`, `CC_PROVIDER`) → provider default.

### 6. The system prompt sets context, then gets out of the way

`prompts.py` is deliberately short. It establishes that the agent is operating
in a real repo and lays down a few hard rules: investigate before acting, make
the smallest correct change, verify with real commands, and prefer dedicated
tools over shelling out. The model is capable; the prompt's job is framing, not
micromanagement.

### 7. The SWE-bench runner attaches around the loop, not inside it

`swebench.py` is the first benchmark system layered on, and it demonstrates the
extension model: it adds a *second entry point* (`cc-swebench`) that drives the
same unchanged `Agent`. Per instance it (1) checks out the instance repo at
`base_commit`, (2) `os.chdir`s in and runs `Agent.run(problem_statement)`, (3)
captures `git diff --cached` against `base_commit` as the predicted patch, and
(4) writes one JSON line (`instance_id`, `model_name_or_path`, `model_patch`).

It deliberately stops at *prediction*. Scoring — building the per-instance
environment, applying the predicted patch plus the gold test patch, running the
hidden tests — is the official `swebench` harness's job, so we don't reimplement
it. Repo clones are cached under `--repos-dir` and reset to a pristine
`base_commit` between instances, so a run is resumable and re-cloning is avoided.

To support this, `Agent` gained two small, loop-shaped additions: it accumulates
`usage` (token totals across every API call) and counts `turns`. The runner
prices those (`PRICING`) into a per-instance USD estimate. Because the loop
re-sends the full transcript every turn and sends no `cache_control`, the
dominant cost is cumulative input tokens — prompt caching / compaction (roadmap)
is the biggest lever to bring it down.

## Data Flow (one task)

```
user task ─▶ cli.main ─▶ Agent.run
                              │
        ┌─────────────────────┴──────────────────────┐
        ▼                                             │
  client.messages.create(system, tools, messages)    │
        │                                             │
        ▼                                             │
  stop_reason == "tool_use" ? ──no──▶ return final text (printed by CLI)
        │ yes                                         │
        ▼                                             │
  for each tool_use: Tool.run() ─▶ tool_result ───────┘
        (ToolError → is_error result, loop continues)
```

## Design Trade-offs

- **Minimalism over features.** No streaming, no sub-agents, no permission
  gating, no context compaction yet. These are deliberately deferred (see the
  roadmap in `README.md`) so the skeleton stays legible.
- **Blocking, single-process.** Simpler to reason about and debug than an async
  or distributed design; adequate for a CLI that drives one model at a time.
- **Model-driven control flow.** Trades predictability for flexibility — the
  agent can tackle open-ended tasks, but its behavior is only as good as the
  model, prompt, and tool descriptions.
- **`max_turns` safety valve.** Caps a runaway trajectory at 60 turns, returning
  a clear "hit max turns" message instead of looping forever.

## Extension Points

- **New capability** → add a `Tool` subclass and register it in
  `default_tools()`.
- **New provider** → add a row to `PROVIDERS` (must speak the Messages API).
- **Behavior change** → edit `SYSTEM_PROMPT`; no code changes needed.
- **New entry point** → drive the `Agent` from a new module (as `swebench.py`
  does) without touching the loop.
- **Future systems** (context compaction, streaming, sub-agents, trajectory
  logging, permission layer) all attach around the same loop without changing
  its shape — the SWE-bench runner is the worked example.
