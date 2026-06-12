"""Smoke tests: the package imports and the tool registry is well-formed.

These don't call the model — they just assert the irreducible skeleton holds
together so CI catches an import error or a malformed tool schema.
"""

from __future__ import annotations

from cc.tools import default_tools


def test_default_tools_registry() -> None:
    tools = default_tools()
    names = {t.name for t in tools}
    assert {"bash", "read", "write", "edit", "glob", "grep"} <= names


def test_every_tool_exposes_a_valid_schema() -> None:
    for tool in default_tools():
        schema = tool.schema()
        assert schema["name"] == tool.name
        assert schema["description"], f"{tool.name} has no description"
        assert schema["input_schema"]["type"] == "object"
