# Context Compaction

This document specifies **context compaction** for *Claude Code, From Scratch* —
a system layered around the loop (see `docs/system-design.md`) that keeps a long
run from overflowing the model's context window.

## Why

`Agent.run` keeps `self.messages` append-only and verbatim (`agent.py`). That is
the right invariant — it preserves the `thinking`/`tool_use` blocks the Messages
API requires — but it means the transcript only grows. On a long task (exactly
the SWE-bench case the project targets) the prompt eventually exceeds the model's
context window and the next `messages.create` call fails, losing the whole run.
There is no recovery today; "no context compaction yet" is called out as a known
gap in `docs/system-design.md` (§Design Trade-offs) and the `README.md` roadmap.

Compaction makes the loop self-healing: when the live context crosses a budget,
the agent spends **one extra model call** to summarize the older turns and
replaces them with that summary, then keeps going.

Two design choices (decided up front):
- **Technique: LLM summarization** — highest fidelity, and what real Claude Code
  does. The alternatives considered and rejected for now were *tool-result
  eviction* (cheap but lossy) and *sliding-window trim* (simplest but drops early
  context blindly).
- **Trigger: token threshold** — driven by the token counts already returned in
  `response.usage`, so it reacts to real content size rather than a blind turn
  count.

## Key constraint

Assistant turns are stored exactly as the API returned them, and **every
`tool_use` block must be paired with its `tool_result`** on the next turn. So
compaction must never delete arbitrary messages. Instead it collapses a *prefix*
of complete turns into one synthesized `user` message and keeps a clean,
self-contained *tail*. The summary itself is produced from a plain-text rendering
of the prefix, which sidesteps block-pairing rules entirely in that call.

## Design

All changes live in `src/cc/agent.py` (the only place that drives the model),
plus one new prompt in `src/cc/prompts.py`. Nothing about the loop's shape
changes — compaction attaches at the top of each iteration.

### Configuration

Per-provider window in the `PROVIDERS` table, overridable by env:

| Key | Source | Default | Meaning |
| --- | --- | --- | --- |
| `context_window` | `PROVIDERS` row / `CC_CONTEXT_WINDOW` | `anthropic`: 200_000, `deepseek`: 1_000_000 | Model's context window in tokens. |
| `compact_threshold` | `CC_COMPACT_THRESHOLD` | `0.8` | Fraction of the window at which to compact. Set `>= 1` to disable. |
| `keep_recent` | `CC_KEEP_RECENT` | `6` | Trailing messages kept verbatim. |

Config precedence stays consistent with the rest of the CLI: CLI/env → provider
default.

### State added to `Agent`

- `self.context_window`, `self.compact_threshold`, `self.keep_recent`
- `self.last_context_tokens = 0` — size of the most recent request's prompt.
- `self.compactions = 0` — how many times we compacted (for reporting; **not**
  counted as agent `turns`).

### Measuring live context

After each response, record the true prompt size (robust to prompt caching even
though it is off today) from the usage fields already summed in `_tally_usage`:

```
last_context_tokens = input_tokens
                    + cache_read_input_tokens
                    + cache_creation_input_tokens
```

### Trigger (top of the loop)

At the **start** of each `run` iteration, before building the request, call
`_maybe_compact()`. It compacts when **both**:

- `last_context_tokens > compact_threshold * context_window`, and
- there is enough history to be worth collapsing
  (`len(messages) > keep_recent + 2`).

The first iteration is a no-op (`last_context_tokens == 0`).

### Compaction step

`_compact()`:

1. **Pick a clean cut.** Start at `cut = len(messages) - keep_recent`, then walk
   `cut` backwards to the nearest index where `messages[cut]["role"] ==
   "assistant"`. This guarantees the retained tail begins on a turn boundary and
   contains no `tool_result` whose `tool_use` was dropped. `messages[:cut]` is
   the prefix to summarize; `messages[cut:]` is kept verbatim.
2. **Render the prefix to text** with a pure helper `_render_transcript`
   (text blocks; `tool_use` name + args; `tool_result` content; long blocks
   truncated). Text-only keeps the summary call free of block-pairing rules.
3. **Summarize.** One `client.messages.create` with a new `COMPACTION_SYSTEM`
   prompt, **no tools**, `max_tokens ≈ 2048`, and `effort="low"` (only when
   `self.extended`). The prompt asks for a structured brief: the original
   task/goal, key facts learned, files inspected/edited and how, current state,
   and next steps. Tally this call's `usage` into `self.usage` so cost accounting
   stays honest; increment `self.compactions` (not `self.turns`).
4. **Rebuild** with a pure helper `_rebuild_after_compaction(task, summary,
   tail)`:

   ```
   [{"role": "user",
     "content": <original task>
                + "\n\n[Earlier conversation compacted to save context. "
                + "Summary of work so far:]\n"
                + summary}]
   + tail
   ```

   The original task is `messages[0]["content"]` (always the first user string).
   The result is valid alternation: `user(summary) → assistant(tail head) → …`.
5. Reset `last_context_tokens = 0` (the next response refreshes it) and `emit` a
   dim progress line, e.g. `· compacted N msgs → summary`.

### Factoring for testability

Following the `tests/` convention that unit tests never call the model, the logic
is split so only one method touches the API:

| Function | Pure? | Responsibility |
| --- | --- | --- |
| `_should_compact()` | yes | threshold + history check |
| `_render_transcript(messages)` | yes | prefix → plain text |
| `_rebuild_after_compaction(task, summary, tail)` | yes | assemble new `messages` |
| `_summarize(text)` | no | the single summary API call |
| `_compact()` / `_maybe_compact()` | no | orchestration |

## Files to change

- `src/cc/agent.py` — provider `context_window`; `__init__` config + state;
  `_maybe_compact` call at the top of `run`; `_compact`, `_summarize`, and the
  pure helpers; update `last_context_tokens` alongside `_tally_usage`.
- `src/cc/prompts.py` — add `COMPACTION_SYSTEM` next to `SYSTEM_PROMPT`.
- `tests/test_compaction.py` (new) — unit tests (below).
- Keep docs in sync (per `CLAUDE.md`): note compaction + the new `CC_*` env vars
  in `CLAUDE.md`; update the `README.md` roadmap; and in
  `docs/system-design.md` move compaction from "deferred" to implemented and link
  here.

## Verification

- `uv run ruff check .` and `uv run pytest` stay green.
- Unit tests (no API):
  - `_should_compact` flips true once `last_context_tokens` crosses
    `threshold * context_window` and there is enough history.
  - cut + `_rebuild_after_compaction`: first message is `user` and contains the
    original task; the tail is byte-for-byte preserved; **no orphaned
    `tool_use`/`tool_result`** across the boundary; roles alternate validly.
  - With `_summarize` stubbed (or a fake client), drive `run` through one
    compaction: `messages` collapses, `compactions == 1`, and the summary call's
    tokens were added to `usage`.
- End-to-end smoke (forces compaction with a tiny window):

  ```bash
  CC_CONTEXT_WINDOW=4000 CC_COMPACT_THRESHOLD=0.5 \
    uv run cc "summarize this repo's architecture, reading several files"
  ```

  Expect a "compacted …" line and a completed run instead of a context-overflow
  error.

## Trade-offs and limits

- **One extra call per compaction.** Summarization costs tokens; `effort="low"`
  and a small `max_tokens` keep it cheap, and it is far cheaper than a failed run.
- **Lossy by nature.** A summary is not the transcript. Keeping the original task
  verbatim and the most recent `keep_recent` turns preserves the immediate
  working set; older detail is compressed.
- **Pathological single turn.** If one tool result alone exceeds the window,
  summarization can't help much — that edge case is better handled by truncating
  giant tool outputs (the rejected *eviction* technique) and is out of scope here.
