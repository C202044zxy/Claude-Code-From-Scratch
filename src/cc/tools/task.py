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