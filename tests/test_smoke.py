"""Smoke tests: the package imports and the tool registry is well-formed.

These don't call the model — they just assert the irreducible skeleton holds
together so CI catches an import error or a malformed tool schema.
"""

from __future__ import annotations

import json

from cc import swebench
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


def test_swebench_cost_estimate() -> None:
    # 1M input + 100k output at Opus 4.8 ($5 / $25 per 1M) = $5 + $2.5.
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    assert swebench.estimate_cost("claude-opus-4-8", usage) == 7.5
    # Unknown model → no price table → 0 rather than a crash.
    assert swebench.estimate_cost("mystery-model", usage) == 0.0


def test_swebench_loads_and_filters_local_jsonl(tmp_path) -> None:
    path = tmp_path / "inst.jsonl"
    rows = [
        {"instance_id": "a-1", "repo": "x/y", "base_commit": "c0"},
        {"instance_id": "b-2", "repo": "x/z", "base_commit": "c1"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    # Filtering preserves the requested order.
    got = swebench.load_instances(str(path), "test", ["b-2", "a-1"])
    assert [i["instance_id"] for i in got] == ["b-2", "a-1"]
