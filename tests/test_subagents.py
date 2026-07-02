"""Unit tests for sub-agents (see docs/subagents.md).

Per the tests/ convention, nothing here calls the real model. Fake child agents
stand in for spawned sub-agents, and the existing FakeClient/FakeStream pattern
(from test_compaction.py) drives the parent loop without network calls.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from cc.agent import Agent
from cc.tools import Tool, default_tools
from cc.tools.task import TaskTool


# --- Fakes -----------------------------------------------------------------

class FakeChildAgent:
    """A canned child agent whose run() returns a fixed string immediately."""

    def __init__(
        self,
        result: str = "child result",
        usage: dict[str, int] | None = None,
        turns: int = 0,
    ) -> None:
        self.result = result
        self.usage = usage or {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        self.turns = turns

    def run(self, prompt: str) -> str:
        return self.result


class SleepingChildAgent(FakeChildAgent):
    """A child whose run() sleeps before returning — for order-preservation tests."""

    def __init__(
        self,
        result: str = "slow child",
        usage: dict[str, int] | None = None,
        turns: int = 0,
        delay: float = 0.2,
    ) -> None:
        super().__init__(result, usage, turns)
        self.delay = delay

    def run(self, prompt: str) -> str:
        time.sleep(self.delay)
        return self.result


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
    """Mimics an SDK content block (attribute access: .type, .text, .name, ...)."""

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
    def __init__(
        self, type: str, text: str | None = None, thinking: str | None = None
    ) -> None:
        self.type = type
        self.text = text
        self.thinking = thinking


class FakeStream:
    """Context manager mimicking client.messages.stream()."""

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
        if not kwargs.get("tools"):
            return self._outer.summary_response
        return self._outer.loop_responses.pop(0)

    def stream(self, **kwargs: Any) -> FakeStream:
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


class FastTool(Tool):
    """A distinct no-op tool with a configurable name for parallelism tests."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"fast tool {name}"
        self.input_schema = {"type": "object", "properties": {}}

    def run(self, **kwargs: Any) -> str:
        return f"{self.name}-result"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Constructing anthropic.Anthropic needs a key present; no call is made.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _make_agent(**kwargs: Any) -> Agent:
    """Create an Agent with silent emit and NoopTool by default."""
    defaults: dict[str, Any] = dict(
        tools=[NoopTool()],
        emit=lambda _msg: None,
    )
    defaults.update(kwargs)
    return Agent(**defaults)


# ===================================================================
# Depth limit
# ===================================================================


def test_depth_zero_parent_gets_task_tool() -> None:
    """At depth 0 (below max_depth=1), the agent gets a task tool."""
    agent = Agent(depth=0, max_depth=1)
    assert "task" in agent.tools_by_name


def test_depth_limit_blocks_task_tool_in_child() -> None:
    """At max_depth, a spawned child must NOT get a task tool — recursion bottoms out."""
    parent = Agent(depth=0, max_depth=1)
    child = parent._spawn_child()
    assert "task" not in child.tools_by_name
    assert child.depth == 1


def test_depth_limit_with_higher_max_depth_allows_grandchildren() -> None:
    """With max_depth=2, children at depth=1 can spawn grandchildren at depth=2."""
    parent = Agent(depth=0, max_depth=2)
    child = parent._spawn_child()
    assert "task" in child.tools_by_name  # child can spawn further
    grandchild = child._spawn_child()
    assert "task" not in grandchild.tools_by_name  # grandchild bottoms out
    assert grandchild.depth == 2


# ===================================================================
# _absorb_children
# ===================================================================


def test_absorb_children_folds_usage_and_turns() -> None:
    """Child usage/turns roll up into the parent's running totals."""
    agent = _make_agent()
    child = FakeChildAgent(
        result="done",
        usage={
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 10,
            "cache_creation_input_tokens": 5,
        },
        turns=3,
    )
    agent._children.append(child)  # type: ignore[arg-type]
    agent._absorb_children()

    assert agent.usage["input_tokens"] == 100
    assert agent.usage["output_tokens"] == 50
    assert agent.usage["cache_read_input_tokens"] == 10
    assert agent.usage["cache_creation_input_tokens"] == 5
    assert agent.turns == 3


def test_absorb_children_handles_multiple_children() -> None:
    """Usage from multiple children sums correctly."""
    agent = _make_agent()
    for _ in range(3):
        child = FakeChildAgent(
            usage={"input_tokens": 10, "output_tokens": 5,
                   "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            turns=1,
        )
        agent._children.append(child)  # type: ignore[arg-type]

    agent._absorb_children()
    assert agent.usage["input_tokens"] == 30
    assert agent.usage["output_tokens"] == 15
    assert agent.turns == 3


def test_absorb_children_is_idempotent_on_empty_list() -> None:
    """Calling _absorb_children with no children is a safe no-op."""
    agent = _make_agent()
    original_usage = dict(agent.usage)
    original_turns = agent.turns
    agent._absorb_children()
    assert agent.usage == original_usage
    assert agent.turns == original_turns


# ===================================================================
# Isolation (integration through run())
# ===================================================================


def test_subagent_result_appears_in_tool_result_only() -> None:
    """The parent sees only the child's final text in the tool_result content.

    The child's intermediate messages live in a separate Agent instance and are
    never visible to the parent — isolation is guaranteed by construction.
    """
    child_result = "Found 3 files: a.py, b.py, c.py"
    child_usage = {
        "input_tokens": 200,
        "output_tokens": 80,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 0,
    }

    agent = _make_agent()  # start with basic NoopTool agent

    def spawn():
        child = FakeChildAgent(result=child_result, usage=child_usage, turns=2)
        agent._children.append(child)  # mirror _spawn_child's registration
        return child

    agent.tools = default_tools(spawn=spawn)
    agent.tools_by_name = {t.name: t for t in agent.tools}

    # First response: the model calls the task tool.
    # Second response: the model finishes after seeing the tool result.
    loop_responses = [
        FakeResponse(
            content=[
                FakeBlock(
                    "tool_use", name="task",
                    input={"description": "find files", "prompt": "Find all .py files"},
                    id="task_1",
                )
            ],
            stop_reason="tool_use",
            usage=FakeUsage(input_tokens=50, output_tokens=30),
        ),
        FakeResponse(
            content=[FakeBlock("text", text="Task complete.")],
            stop_reason="end_turn",
            usage=FakeUsage(input_tokens=10, output_tokens=5),
        ),
    ]
    summary_response = FakeResponse(
        [FakeBlock("text", text="")], "end_turn", FakeUsage()
    )
    agent.client = FakeClient(loop_responses, summary_response)

    result = agent.run("use a sub-agent to find files")

    # The parent's final text comes from the second response.
    assert result == "Task complete."

    # The tool_result must contain the child's final text.
    tool_result_msgs = [
        m for m in agent.messages
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert len(tool_result_msgs) == 1
    tool_results = tool_result_msgs[0]["content"]
    task_result = [tr for tr in tool_results if tr["tool_use_id"] == "task_1"]
    assert len(task_result) == 1
    assert task_result[0]["content"] == child_result
    assert task_result[0]["is_error"] is False

    # The parent's messages must NOT contain any of the child's intermediate data.
    # (This is trivially true — the child is a separate Agent instance — but we
    # verify by checking that the child's inner messages aren't in the parent.)
    all_parent_content = str(agent.messages)
    assert "FakeChildAgent" not in all_parent_content  # child itself not serialized

    # Child usage must fold into the parent's running totals.
    # Parent's own API calls: (50+30) + (10+5) = 95 input
    # Child's usage: 200 input
    assert agent.usage["input_tokens"] == 50 + 10 + 200
    assert agent.usage["output_tokens"] == 30 + 5 + 80
    assert agent.usage["cache_read_input_tokens"] == 20
    # Parent turns: 2 (loop responses) + 2 (child turns) = 4
    assert agent.turns == 4


def test_subagent_error_propagates_as_tool_error() -> None:
    """If a child's run() raises, the exception becomes an is_error tool_result."""

    class FailingChild(FakeChildAgent):
        def run(self, prompt: str) -> str:
            raise RuntimeError("child crashed")

    def spawn():
        return FailingChild()

    tools = default_tools(spawn=spawn)
    agent = _make_agent(tools=tools)

    loop_responses = [
        FakeResponse(
            content=[
                FakeBlock(
                    "tool_use", name="task",
                    input={"description": "doomed", "prompt": "crash"},
                    id="task_err",
                )
            ],
            stop_reason="tool_use",
            usage=FakeUsage(input_tokens=10, output_tokens=5),
        ),
        FakeResponse(
            content=[FakeBlock("text", text="Handled error.")],
            stop_reason="end_turn",
            usage=FakeUsage(input_tokens=5, output_tokens=3),
        ),
    ]
    summary_response = FakeResponse(
        [FakeBlock("text", text="")], "end_turn", FakeUsage()
    )
    agent.client = FakeClient(loop_responses, summary_response)

    result = agent.run("doomed task")
    assert result == "Handled error."

    # The task tool_result must be marked is_error.
    tool_result_msgs = [
        m for m in agent.messages
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    task_result = [
        tr for tr in tool_result_msgs[0]["content"]
        if tr["tool_use_id"] == "task_err"
    ][0]
    assert task_result["is_error"] is True
    assert "RuntimeError" in task_result["content"]
    assert "child crashed" in task_result["content"]


def test_subagent_unknown_tool_returns_error() -> None:
    """Calling an unregistered tool name returns an is_error result."""
    agent = _make_agent()
    # Directly test _execute (bypasses run loop).
    result, is_error = agent._execute("nonexistent_tool", {})
    assert is_error is True
    assert "Unknown tool" in result


# ===================================================================
# Tool result order preservation under parallelism
# ===================================================================


def test_tool_results_preserve_order_with_slow_task() -> None:
    """Results maintain input order even when a slow task completes last.

    Three tool_use blocks: fast1, task (slow child sleeps 0.2s), fast2.
    The task result must appear in position 1 (0-indexed), between the two
    fast results, despite completing after both fast tools finished.
    """
    fast1 = FastTool("fast1")
    fast2 = FastTool("fast2")
    agent = _make_agent(tools=[fast1, fast2])  # placeholder, replaced below

    def slow_spawn():
        child = SleepingChildAgent(
            result="slow child done",
            usage={"input_tokens": 5, "output_tokens": 2,
                   "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
            turns=1,
            delay=0.2,
        )
        agent._children.append(child)  # mirror _spawn_child's registration
        return child

    task_tool = TaskTool(slow_spawn)
    agent.tools = [fast1, task_tool, fast2]
    agent.tools_by_name = {t.name: t for t in agent.tools}

    blocks = [
        FakeBlock("tool_use", name="fast1", input={}, id="b1"),
        FakeBlock("tool_use", name="task",
                  input={"description": "slow", "prompt": "take your time"}, id="b2"),
        FakeBlock("tool_use", name="fast2", input={}, id="b3"),
    ]

    results = agent._execute_tools(blocks)
    agent._absorb_children()  # normally called by run() after _execute_tools

    assert len(results) == 3
    # Order must match input order regardless of completion time.
    assert results[0]["tool_use_id"] == "b1"
    assert results[0]["content"] == "fast1-result"
    assert results[0]["is_error"] is False

    assert results[1]["tool_use_id"] == "b2"
    assert results[1]["content"] == "slow child done"
    assert results[1]["is_error"] is False

    assert results[2]["tool_use_id"] == "b3"
    assert results[2]["content"] == "fast2-result"
    assert results[2]["is_error"] is False

    # Child usage must have been absorbed even in the parallel path.
    assert agent.usage["input_tokens"] == 5
    assert agent.usage["output_tokens"] == 2
    assert agent.turns == 1


def test_single_tool_uses_fast_path() -> None:
    """A single tool_use block skips the thread pool entirely."""
    agent = _make_agent()
    blocks = [FakeBlock("tool_use", name="noop", input={}, id="only")]
    results = agent._execute_tools(blocks)
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "only"
    assert results[0]["content"] == "ok"


# ===================================================================
# Tool registry
# ===================================================================


def test_default_tools_excludes_task_when_spawn_is_none() -> None:
    """Without a spawn callback, the task tool is absent from the registry."""
    tools = default_tools(spawn=None)
    names = {t.name for t in tools}
    assert "task" not in names
    assert {"bash", "read", "write", "edit", "glob", "grep"} <= names


def test_default_tools_includes_task_when_spawn_provided() -> None:
    """When a spawn callback is given, the task tool appears in the registry."""
    tools = default_tools(spawn=lambda: FakeChildAgent())
    names = {t.name for t in tools}
    assert "task" in names
    assert {"bash", "read", "write", "edit", "glob", "grep", "task"} <= names


# ===================================================================
# Compaction: children do NOT affect parent compaction threshold
# ===================================================================


def test_child_usage_does_not_trigger_parent_compaction() -> None:
    """Children have their own context window; their tokens don't count toward
    the parent's compaction threshold (which reads last_context_tokens from
    the parent's own API responses, not from self.usage)."""
    agent = _make_agent(context_window=1000, compact_threshold=0.5)

    # Simulate a child with huge usage — this goes into self.usage but NOT
    # into last_context_tokens (which comes from API responses).
    child = FakeChildAgent(
        usage={"input_tokens": 999_999, "output_tokens": 999_999,
               "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        turns=50,
    )
    agent._children.append(child)  # type: ignore[arg-type]
    agent._absorb_children()

    # Usage reflects the child's tokens...
    assert agent.usage["input_tokens"] == 999_999
    # ...but last_context_tokens is still 0 (no API response processed).
    assert agent.last_context_tokens == 0
    # So compaction should NOT trigger.
    agent.messages = [{"role": "user", "content": "x"}] * 10
    assert agent._should_compact() is False
