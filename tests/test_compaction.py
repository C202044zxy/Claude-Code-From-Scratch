"""Unit tests for context compaction (see docs/context-compaction.md).

Per the tests/ convention, nothing here calls the real model. The pure helpers
are exercised directly; the one method that touches the API (`_summarize`) is
driven through a fake client so we can run `Agent.run` across a real compaction
without a network call.
"""

from __future__ import annotations

from typing import Any

import pytest

from cc.agent import Agent, _context_tokens
from cc.tools.base import Tool


# --- Fakes -----------------------------------------------------------------


class FakeUsage:
    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


class FakeBlock:
    """Mimics an SDK content block (attribute access: .type, .text, ...)."""

    def __init__(
        self,
        type: str,
        text: str | None = None,
        name: str | None = None,
        input: dict[str, Any] | None = None,
        id: str | None = None,
    ) -> None:
        self.type = type
        self.text = text
        self.name = name
        self.input = input
        self.id = id


class FakeResponse:
    def __init__(
        self, content: list[FakeBlock], stop_reason: str, usage: FakeUsage
    ) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage


class FakeStreamEvent:
    """Mimics an SDK enriched stream event (.type + .text / .thinking)."""

    def __init__(
        self, type: str, text: str | None = None, thinking: str | None = None
    ) -> None:
        self.type = type
        self.text = text
        self.thinking = thinking


class FakeStream:
    """Context manager mimicking `client.messages.stream(...)`.

    Iterating it yields one event per text/thinking block in the final
    response; `get_final_message()` hands back that same FakeResponse, so the
    loop's downstream handling (usage, stop_reason, append) is unchanged.
    """

    def __init__(self, response: FakeResponse) -> None:
        self._response = response

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def __iter__(self):
        for block in self._response.content:
            if block.type == "text":
                yield FakeStreamEvent("text", text=block.text)
            elif block.type == "thinking":
                yield FakeStreamEvent("thinking", thinking=block.text)

    def get_final_message(self) -> FakeResponse:
        return self._response


class FakeMessages:
    def __init__(self, outer: "FakeClient") -> None:
        self._outer = outer

    def create(self, **kwargs: Any) -> FakeResponse:
        self._outer.calls.append(kwargs)
        # A summary call carries no tools; a loop turn always does.
        if not kwargs.get("tools"):
            return self._outer.summary_response
        return self._outer.loop_responses.pop(0)

    def stream(self, **kwargs: Any) -> FakeStream:
        # Loop turns flow through stream(); record the call like create() does
        # so call-tracking assertions still see them.
        self._outer.calls.append(kwargs)
        return FakeStream(self._outer.loop_responses.pop(0))


class FakeClient:
    def __init__(
        self, loop_responses: list[FakeResponse], summary_response: FakeResponse
    ) -> None:
        self.loop_responses = loop_responses
        self.summary_response = summary_response
        self.calls: list[dict[str, Any]] = []
        self.messages = FakeMessages(self)


class NoopTool(Tool):
    name = "noop"
    description = "does nothing"
    input_schema = {"type": "object", "properties": {}}

    def run(self, **kwargs: Any) -> str:
        return "ok"


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Constructing anthropic.Anthropic needs a key present; no call is made.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def make_agent(**kwargs: Any) -> Agent:
    defaults: dict[str, Any] = dict(
        tools=[NoopTool()],
        context_window=1000,
        compact_threshold=0.5,
        keep_recent=2,
        emit=lambda _msg: None,
    )
    defaults.update(kwargs)
    return Agent(**defaults)


# --- _context_tokens -------------------------------------------------------


def test_context_tokens_sums_all_input_fields() -> None:
    usage = FakeUsage(
        input_tokens=100,
        output_tokens=999,  # output must NOT count toward context size
        cache_read_input_tokens=20,
        cache_creation_input_tokens=3,
    )
    resp = FakeResponse([], "end_turn", usage)
    assert _context_tokens(resp) == 123


def test_context_tokens_handles_missing_usage() -> None:
    class NoUsage:
        usage = None

    assert _context_tokens(NoUsage()) == 0


# --- _should_compact -------------------------------------------------------


def test_should_compact_false_below_threshold() -> None:
    agent = make_agent()
    agent.messages = [{"role": "user", "content": "x"}] * 10
    agent.last_context_tokens = 499  # threshold = 0.5 * 1000 = 500
    assert agent._should_compact() is False


def test_should_compact_true_when_over_threshold_and_enough_history() -> None:
    agent = make_agent()
    agent.messages = [{"role": "user", "content": "x"}] * 10
    agent.last_context_tokens = 501
    assert agent._should_compact() is True


def test_should_compact_false_without_enough_history() -> None:
    agent = make_agent()  # keep_recent=2 → need len > 4
    agent.messages = [{"role": "user", "content": "x"}] * 4
    agent.last_context_tokens = 900
    assert agent._should_compact() is False


def test_should_compact_false_on_first_iteration() -> None:
    agent = make_agent()
    agent.messages = [{"role": "user", "content": "x"}] * 10
    # last_context_tokens defaults to 0 before any response.
    assert agent._should_compact() is False


# --- _render_transcript ----------------------------------------------------


def test_render_transcript_covers_block_kinds_and_skips_thinking() -> None:
    agent = make_agent()
    messages = [
        {"role": "user", "content": "the original task"},
        {
            "role": "assistant",
            "content": [
                FakeBlock("thinking", text="secret reasoning"),
                FakeBlock("text", text="let me read a file"),
                FakeBlock("tool_use", name="read", input={"path": "a.py"}),
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "1",
                    "content": "file contents",
                    "is_error": False,
                }
            ],
        },
    ]
    text = agent._render_transcript(messages)
    assert "the original task" in text
    assert "let me read a file" in text
    assert "[tool_use read(" in text
    assert "file contents" in text
    # thinking content must never reach the summarizer.
    assert "secret reasoning" not in text


# --- cut + _rebuild_after_compaction ---------------------------------------


def test_rebuild_preserves_task_and_tail_and_alternation() -> None:
    agent = make_agent()
    task = "fix the bug in foo.py"
    tail = [
        {"role": "assistant", "content": [FakeBlock("text", text="recent turn")]},
        {"role": "user", "content": "follow up"},
    ]
    rebuilt = agent._rebuild_after_compaction(task, "SUMMARY", tail)

    # First message is a user turn carrying the original task and the summary.
    assert rebuilt[0]["role"] == "user"
    assert task in rebuilt[0]["content"]
    assert "SUMMARY" in rebuilt[0]["content"]
    # Tail is preserved byte-for-byte (same objects).
    assert rebuilt[1:] == tail
    # Roles alternate validly across the seam: user -> assistant -> ...
    roles = [m["role"] for m in rebuilt]
    assert roles == ["user", "assistant", "user"]


def test_cut_lands_on_assistant_boundary_no_orphaned_blocks() -> None:
    agent = make_agent(keep_recent=2)
    # A realistic transcript: task, then two tool-use round trips.
    agent.messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": [FakeBlock("tool_use", name="noop", id="a")]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}],
        },
        {"role": "assistant", "content": [FakeBlock("tool_use", name="noop", id="b")]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "b", "content": "ok"}],
        },
    ]
    cut = len(agent.messages) - agent.keep_recent
    while cut > 0 and agent.messages[cut]["role"] != "assistant":
        cut -= 1
    # The tail must begin on an assistant turn (so its tool_use is intact) and
    # must not start with a tool_result whose tool_use was cut away.
    tail = agent.messages[cut:]
    assert tail[0]["role"] == "assistant"
    first_block = tail[0]["content"][0]
    assert first_block.type == "tool_use"


# --- run() through one real compaction -------------------------------------


def test_run_compacts_once_and_tallies_summary_tokens() -> None:
    agent = make_agent()

    def tool_use_turn(input_tokens: int, block_id: str) -> FakeResponse:
        return FakeResponse(
            content=[FakeBlock("tool_use", name="noop", input={}, id=block_id)],
            stop_reason="tool_use",
            usage=FakeUsage(input_tokens=input_tokens, output_tokens=5),
        )

    loop_responses = [
        tool_use_turn(10, "a"),  # small: no compaction next iter
        tool_use_turn(600, "b"),  # > 500 threshold: triggers compaction
        FakeResponse(
            content=[FakeBlock("text", text="all done")],
            stop_reason="end_turn",
            usage=FakeUsage(input_tokens=5),
        ),
    ]
    summary_response = FakeResponse(
        content=[FakeBlock("text", text="THE SUMMARY")],
        stop_reason="end_turn",
        usage=FakeUsage(input_tokens=7, output_tokens=11),
    )
    agent.client = FakeClient(loop_responses, summary_response)

    result = agent.run("do the thing")

    assert result == "all done"
    assert agent.compactions == 1
    # The summary call's tokens were folded into the running usage totals.
    assert agent.usage["output_tokens"] >= 11
    assert agent.usage["input_tokens"] >= 7
    # After compaction the transcript collapsed to: rebuilt user(summary) + the
    # kept tail (2) + the final assistant turn from the last loop call.
    assert len(agent.messages) == 4
    assert agent.messages[0]["role"] == "user"
    assert "THE SUMMARY" in agent.messages[0]["content"]
    assert "do the thing" in agent.messages[0]["content"]
    # A summary call (no tools) happened exactly once.
    summary_calls = [c for c in agent.client.calls if not c.get("tools")]
    assert len(summary_calls) == 1
    assert summary_calls[0]["system"] == agent.compaction_prompt


# --- streaming -------------------------------------------------------------


def test_run_streams_thinking_and_text_deltas() -> None:
    captured: list[str] = []
    agent = make_agent(stream_to=captured.append)

    response = FakeResponse(
        content=[
            FakeBlock("thinking", text="let me think"),
            FakeBlock("text", text="here is the answer"),
        ],
        stop_reason="end_turn",
        usage=FakeUsage(input_tokens=5),
    )
    summary = FakeResponse([FakeBlock("text", text="")], "end_turn", FakeUsage())
    agent.client = FakeClient([response], summary)

    result = agent.run("do the thing")

    assert result == "here is the answer"
    out = "".join(captured)
    # Both deltas streamed live, thinking wrapped in the dim ANSI codes.
    assert "let me think" in out
    assert "here is the answer" in out
    assert "\033[2m" in out  # dim on (thinking)
    assert "\033[0m" in out  # dim off (thinking → text transition)
