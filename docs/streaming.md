# Streaming output — token-level visibility during long turns

## Why

The agent loop blocks on `self.client.messages.create(...)` (`src/cc/agent.py`) and only
renders text **after** the full response arrives (`_render_assistant`). With adaptive
thinking + `effort: high`, a single turn can run for many seconds with **zero feedback** —
the user stares at a blank terminal until the whole turn lands. Streaming prints thinking
and text deltas as the model generates them.

The Anthropic SDK ships exactly the tool for this: `client.messages.stream(...)` is a
context manager that yields token deltas as enriched events **and** exposes
`get_final_message()`, which returns the same `Message` object shape as `.create()`. So
usage tallying (`_tally_usage`), context-token tracking (`_context_tokens`), the
`stop_reason` check, and the **verbatim-append invariant** all keep working untouched — we
only change *how* we obtain the response and add live rendering.

DeepSeek speaks the same Anthropic-compatible API (only `base_url` differs), so one
streaming path covers both providers.

Design decisions:
- **Stream thinking + text.** Thinking deltas render dimmed; text renders normally. Long
  turns are mostly thinking, so that's where the visibility matters most.
- **Always-on.** The main loop always streams; *visibility* is controlled by an output
  sink, not a global toggle. No new env var or CLI flag.

## Approach

### 1. A second output sink for partial (no-newline) writes — `src/cc/agent.py`

The existing `emit: Emit = print` is line-oriented (one newline per call) and is used for
discrete progress lines (tool calls, compaction). Token streaming needs raw partial writes
with a flush per token. Add a parallel sink rather than overloading `emit`:

```python
import sys

def _stdout_write(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()
```

Add constructor param `stream_to: Emit = _stdout_write` stored as `self.stream_to`. Keep
`emit` exactly as-is for discrete lines. Tests / SWE-bench can pass a capturing or no-op
`stream_to`.

### 2. Stream in the main loop — `run()`

Replace the `.create(...)` call + `_render_assistant(response.content)` with a streamed
call that renders deltas live, then pulls the final message:

```python
with self.client.messages.stream(
    model=self.model,
    max_tokens=self.max_tokens,
    system=self.system,
    tools=[t.schema() for t in self.tools],
    messages=self.messages,
    **extra,
) as stream:
    self._render_stream(stream)
    response = stream.get_final_message()

self.turns += 1
self._tally_usage(response)
self.last_context_tokens = _context_tokens(response)
self.messages.append({"role": "assistant", "content": response.content})
```

Everything downstream (`stop_reason`, `_final_text`, tool execution, message append) is
unchanged. `_render_assistant` is no longer called from the loop — remove it.

### 3. Live renderer — new `_render_stream`

Iterate the SDK's enriched stream events. The two we care about:
- `event.type == "text"` → `event.text` is the text delta.
- `event.type == "thinking"` → `event.thinking` is the thinking delta.

Render thinking dimmed (ANSI `\033[2m` … `\033[0m`, matching the dim used for tool/
compaction lines), switching cleanly on the thinking→text transition, and terminate the
streamed block with a single trailing newline so the subsequent line-oriented `emit`
(tool-call lines) starts cleanly:

```python
def _render_stream(self, stream: Any) -> None:
    in_thinking = False
    wrote_any = False
    for event in stream:
        if event.type == "thinking":
            if not in_thinking:
                self.stream_to("\033[2m")  # dim on
                in_thinking = True
            self.stream_to(event.thinking)
            wrote_any = True
        elif event.type == "text":
            if in_thinking:
                self.stream_to("\033[0m\n")  # close thinking block
                in_thinking = False
            self.stream_to(event.text)
            wrote_any = True
    if in_thinking:
        self.stream_to("\033[0m")
    if wrote_any:
        self.stream_to("\n")
```

Notes:
- The final return value still goes through `_final_text` (stripped) — unchanged.
- `_summarize` stays on `.create()`: its output is folded into the compaction summary,
  never shown to the user, and it's small (2048 tokens). No reason to stream it, and
  leaving it avoids touching the summary path.

### 4. Keep SWE-bench quiet — `src/cc/swebench.py`

SWE-bench runs non-interactively and captures only the final patch + a summary line; it
already injects its own `emit`. Pass `stream_to=lambda _t: None` to the `Agent(...)`
construction so token deltas don't spam the run logs. Its existing tool/compaction `emit`
lines are unaffected.

### 5. CLI — `src/cc/cli.py`

No change required: the default `stream_to=_stdout_write` writes tokens to stdout in real
time, which is exactly what the interactive CLI wants. The existing `emit=print` default
continues to handle tool/compaction lines.

## Tests — `tests/test_compaction.py`

The fakes mock `client.messages.create`; the loop now calls `client.messages.stream`. Add
a fake streaming path while preserving the existing assertions:

- Add `FakeStream` (context manager): `__enter__`/`__exit__`, `__iter__` yielding fake
  events derived from the response's content blocks (a `text` / `thinking` event per
  matching block), and `get_final_message()` returning the `FakeResponse`.
- Add `FakeMessages.stream(**kwargs)` that appends kwargs to `self._outer.calls` (so call
  tracking still works), pops from `loop_responses`, and returns a `FakeStream`. Keep
  `create()` for the summary call.
- `test_run_compacts_once_and_tallies_summary_tokens` keeps passing: loop turns now flow
  through `stream()` (still recorded in `calls`), the summary call still flows through
  `create()` with no `tools`, so the `summary_calls` filter still finds exactly one.
- Add one focused test: construct an Agent with a capturing `stream_to`, drive a single
  `end_turn` response whose content has a thinking block + a text block, and assert the
  captured stream output contains both deltas (and the dim codes around thinking).

Run `uv run ruff check .` and `uv run pytest` (CI-equivalent) before committing.

## Docs to keep in sync

- `README.md` and `CLAUDE.md`: note that the loop streams via `messages.stream()` and
  renders thinking/text token-by-token; update the `agent.py` description (it currently
  says the loop calls `messages.create`).
- This file (`docs/streaming.md`).

## Verification

1. `uv run cc "summarize what src/cc/agent.py does"` against this repo — watch thinking
   stream dimmed, then the answer stream in real time; confirm tool-call lines still render
   correctly between turns and there's no double-printing of the final text.
2. `uv run cc --provider deepseek "..."` (if `DEEPSEEK_API_KEY` is set) — confirm streaming
   works without `thinking`/`output_config` (provider `extended=False` path).
3. `uv run cc-swebench --limit 1` — confirm logs are NOT flooded with token deltas and the
   per-instance summary line + patch are unchanged.
4. `uv run ruff check .` and `uv run pytest` both green.
