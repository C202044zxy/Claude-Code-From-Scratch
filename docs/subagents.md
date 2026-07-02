# Sub-agents — a `task` tool that spawns a nested loop

## Why

Two pressures push on the single flat loop as tasks get bigger:

1. **Context pollution.** Exploratory sub-tasks ("find everywhere the retry logic
   lives", "figure out why `test_foo` fails") can burn dozens of tool calls and tens of
   thousands of tokens of `grep`/`read` output — almost all of it scaffolding the parent
   never needs again. It sits in `self.messages` until compaction evicts it, crowding out
   the tokens that matter.
2. **No parallelism.** The loop runs one turn at a time and, within a turn, executes
   `tool_use` blocks one after another (`agent.py` ~L169–183). Three independent
   investigations that could run at once instead run back to back.

A sub-agent solves both. It's a **fresh `Agent` loop** run to completion on a focused
prompt, returning only its **final text** to the parent as a `tool_result`. The parent's
context absorbs a single synthesized answer instead of the whole investigation, and
because each sub-agent is an independent, I/O-bound loop, several launched in one turn can
run concurrently.

This is the same shape as Claude Code's `Task` tool, and it falls out of the existing
design almost for free: a sub-agent *is* an `Agent`, and "add a capability = append a
`Tool`". The only genuinely new pieces are (a) giving one tool the ability to construct a
child `Agent` without a circular import, and (b) letting a turn's tool calls run in
parallel.

Design decisions:
- **The parent sees only the final message.** Not the sub-agent's thinking, tool calls, or
  intermediate results. That isolation is the whole point — it's what keeps the parent's
  context clean. The cost: the prompt to the sub-agent must be self-contained, and we must
  ask it for a complete report (the `description` of the tool teaches the model this).
- **Depth-limited to 1 by default.** A sub-agent does *not* get its own `task` tool, so it
  cannot spawn grandchildren. This is the cheap guard against runaway fork-bombs and is
  enough for every real use; deeper nesting can come later behind `max_depth`.
- **Sub-agents are quiet.** They run with no-op `emit`/`stream_to`, so concurrent loops
  don't interleave token streams into the terminal. The parent emits one progress line per
  sub-agent. (This reuses the exact mechanism SWE-bench already uses to stay silent.)
- **Inherit the parent's config.** Same model, provider, effort, max_tokens, and
  compaction settings. A sub-agent is the same agent pointed at a narrower job.

## Approach

### 1. The `task` tool — `src/cc/tools/task.py` (new)

A `Tool` whose `run()` builds a child `Agent`, runs it to completion, and returns its final
text. It can't import `Agent` (that would make `agent.py → tools → agent.py` circular), so
it receives a **`spawn` callback** that produces a ready-to-run child. The tool stays
ignorant of how children are constructed.

```python
from __future__ import annotations

from typing import Any, Callable

from .base import Tool

# Callable that returns a fresh, ready-to-run child Agent (typed loosely to avoid
# importing Agent here — that would be circular).
Spawn = Callable[[], Any]


class TaskTool(Tool):
    name = "task"
    description = (
        "Spawn a sub-agent to handle a focused, self-contained sub-task and return "
        "ONLY its final result. Use this to (a) parallelize independent work — launch "
        "several task calls in a single turn and they run concurrently — or (b) keep "
        "your own context clean when a sub-task needs heavy exploration (e.g. 'find "
        "everywhere X is configured', 'investigate why test Y fails'). The sub-agent "
        "starts FRESH: it has no memory of this conversation and sees only the `prompt` "
        "you pass, and you receive only its final text answer, not its intermediate "
        "steps. So put everything it needs in `prompt` and ask for a specific, complete "
        "report. The sub-agent cannot spawn further sub-agents. Prefer a direct tool "
        "call for quick single-step actions; use a sub-agent when the work is "
        "substantial or parallelizable."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "A short (3-5 word) label for this sub-task.",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "The complete, self-contained task for the sub-agent. It sees only "
                    "this — include all needed context and state exactly what to report "
                    "back."
                ),
            },
        },
        "required": ["description", "prompt"],
    }

    def __init__(self, spawn: Spawn) -> None:
        self._spawn = spawn

    def run(self, description: str, prompt: str) -> str:
        child = self._spawn()
        return child.run(prompt)
```

`description` is unused by `run()` but is what the model writes to name the work; the
parent surfaces it in its progress line (see §3, via `_fmt_args`). Keeping it in the schema
also nudges the model to think about scoping the sub-task.

### 2. Wire spawning into the Agent — `src/cc/agent.py`

`default_tools()` gains an optional `spawn`; the `task` tool is appended only when one is
provided:

```python
# tools/__init__.py
from .task import TaskTool

def default_tools(spawn=None) -> list[Tool]:
    tools = [BashTool(), ReadTool(), WriteTool(), EditTool(), GlobTool(), GrepTool()]
    if spawn is not None:
        tools.append(TaskTool(spawn))
    return tools
```

`Agent.__init__` gets two new params and a children list, set **before** it builds tools
(the spawn closure reads `self.depth`/`self.max_depth`):

```python
def __init__(self, ..., depth: int = 0, max_depth: int | None = None) -> None:
    ...
    self.depth = depth
    self.max_depth = max_depth if max_depth is not None else int(
        os.getenv("CC_MAX_DEPTH", 1)
    )
    self._children: list["Agent"] = []
    # A child can spawn only if we're still below the depth limit; otherwise the
    # child gets no `task` tool and the recursion bottoms out.
    spawn = self._spawn_child if self.depth < self.max_depth else None
    self.tools = tools if tools is not None else default_tools(spawn=spawn)
    self.tools_by_name = {t.name: t for t in self.tools}
    ...
```

The factory builds a child that inherits config but runs silent and one level deeper:

```python
def _spawn_child(self) -> "Agent":
    child = Agent(
        system=SUBAGENT_PROMPT,
        model=self.model,
        max_tokens=self.max_tokens,
        effort=self.effort,
        provider=self.provider,
        emit=lambda _t: None,
        stream_to=lambda _t: None,
        max_turns=self.max_turns,
        compaction_prompt=self.compaction_prompt,
        context_window=self.context_window,
        compact_threshold=self.compact_threshold,
        keep_recent=self.keep_recent,
        depth=self.depth + 1,
        max_depth=self.max_depth,
    )
    self._children.append(child)
    return child
```

> Note: `Agent.__init__` re-reads provider config from env, but every field that has a
> CLI/env origin is passed explicitly here, so the child faithfully mirrors the parent
> regardless of ambient env.

### 3. Run a turn's tool calls in parallel — `agent.py::run()`

Today the loop iterates `tool_use` blocks sequentially. Replace that with a bounded thread
pool, **preserving result order** so each `tool_result` still pairs with its
`tool_use_id`. Tool calls are I/O-bound (API calls, subprocess, disk), so threads give real
concurrency despite the GIL — and multiple `task` calls in one turn now run at the same
time:

```python
from concurrent.futures import ThreadPoolExecutor

blocks = [b for b in response.content if b.type == "tool_use"]
if len(blocks) == 1:
    results = [self._execute(blocks[0].name, blocks[0].input)]
else:
    with ThreadPoolExecutor(max_workers=min(len(blocks), 8)) as ex:
        results = list(ex.map(lambda b: self._execute(b.name, b.input), blocks))

tool_results = [
    {
        "type": "tool_result",
        "tool_use_id": b.id,
        "content": result,
        "is_error": is_error,
    }
    for b, (result, is_error) in zip(blocks, results)
]
self.messages.append({"role": "user", "content": tool_results})
self._absorb_children()  # fold child usage/turns into our running totals
```

`ex.map` preserves input order, so `zip(blocks, results)` stays aligned. The single-block
fast path keeps the common case allocation-free and easy to read.

**Trade-off — concurrent file mutation.** Parallelizing *all* tools (not just `task`) means
two `edit`/`write`/`bash` calls in the same turn run concurrently and could race on the
filesystem. In practice the model batches independent calls (parallel reads, parallel
sub-agents) and this matches real Claude Code's behavior, but it's a real change in
semantics. If we want to be conservative, gate parallelism on the turn being *all* `task`
calls and run anything else sequentially — start simple (parallelize all), tighten only if
a race shows up.

### 4. Fold child usage back — `agent.py`

A sub-agent's tokens are real spend; SWE-bench prices `agent.usage`, so child usage must
roll up. Children accumulate their own totals; after the pool joins (back on the main
thread — no lock needed) the parent absorbs them:

```python
def _absorb_children(self) -> None:
    while self._children:
        child = self._children.pop()
        for key in self.usage:
            self.usage[key] += child.usage[key]
        self.turns += child.turns
```

`list.append` (in `_spawn_child`, possibly from worker threads) and `list.pop` (here, main
thread after join) are individually atomic under the GIL, and the two phases never overlap,
so this is safe without locking.

### 5. Sub-agent system prompt — `src/cc/prompts.py`

The default `SYSTEM_PROMPT` ends with "give a brief summary"; a sub-agent's *entire* value
is that final message, so it needs a stronger instruction. Add a focused variant:

```python
SUBAGENT_PROMPT = SYSTEM_PROMPT + """

You are a sub-agent: a focused worker spawned to complete one specific task. You have no
memory of any larger conversation and the agent that spawned you will see ONLY your final
message — not your reasoning, tool calls, or intermediate findings. Therefore your final
message must be a complete, self-contained report: state what you found or did, include the
concrete details the caller asked for (file paths, names, results, the actual answer), and
do not refer to context the caller cannot see.
"""
```

### 6. CLI / config — no required change

`max_depth` defaults to 1 via `CC_MAX_DEPTH` (§2), so the `cc` CLI gets sub-agents
automatically. Optionally expose `--max-depth` in `cli.py` for symmetry with `--model`
etc., but it's not needed for the feature.

## Tests — `tests/test_subagents.py` (new)

Reuse the fake-client pattern from `tests/test_compaction.py` (the `FakeStream`/
`FakeMessages` doubles). Cover:

- **Isolation.** Drive a parent whose first turn issues one `task` `tool_use`, then ends.
  Stub the child via a `spawn` that returns a fake agent whose `run()` yields a canned
  string. Assert the parent's `messages` contain exactly that string in the `tool_result`
  and **none** of the child's intermediate blocks.
- **Depth limit.** Construct `Agent(depth=0, max_depth=1)` and assert `"task"` is in
  `tools_by_name`; construct the child it spawns (`depth=1`) and assert `"task"` is
  **absent** — recursion bottoms out.
- **Usage folds up.** Give the fake child non-zero `usage`/`turns`; after a turn that
  spawns it, assert the parent's totals increased by the child's (verifies
  `_absorb_children`).
- **Order preserved under parallelism.** One turn with three `tool_use` blocks (mixed
  `read`/`task`); assert the three `tool_result`s come back in the same order as the
  blocks, each matched to its `tool_use_id`. A child whose `run()` sleeps briefly before
  returning makes the ordering assertion meaningful.

Run `uv run ruff check .` and `uv run pytest` (CI-equivalent) before committing.

## Docs to keep in sync

- `README.md` and `CLAUDE.md`: note the `task` sub-agent tool in the tools list, that a
  turn's tool calls now run in parallel, and the new `CC_MAX_DEPTH` config. Update the
  `agent.py` description (it currently says tools execute in one sequential pass).
- This file (`docs/subagents.md`).

## Verification

1. `uv run cc "Use sub-agents to investigate, in parallel, (a) how the agent loop streams
   output and (b) how compaction decides when to fire, then summarize both."` — confirm the
   parent emits a `task(...)` line per sub-agent, the two run without interleaving garbled
   output, and the final answer reflects both investigations.
2. `uv run cc "summarize what src/cc/agent.py does"` — a task with no sub-agents still works
   unchanged (single-block fast path).
3. `uv run cc-swebench --limit 1` — logs stay quiet (sub-agents inherit no-op sinks) and the
   per-instance cost line still prints, now including any sub-agent tokens.
4. `uv run ruff check .` and `uv run pytest` both green.
```
