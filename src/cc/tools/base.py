"""The Tool abstraction.

A tool is the unit that gives the agent leverage over the world. Each tool is:
  - a `name` the model calls,
  - a `description` the model reads to decide *when* to call it,
  - an `input_schema` (JSON Schema) describing its arguments,
  - a `run()` that actually executes and returns a string result.

The agent loop never hard-codes any tool; it just iterates over a registry and
hands their schemas to the model. Adding a capability = adding a Tool subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ToolError(Exception):
    """Raised by a tool when execution fails in an expected way.

    The agent loop catches this and feeds the message back to the model as a
    tool_result with is_error=True, so the model can recover rather than crash.
    """


class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool. Return a string result, or raise ToolError."""

    def schema(self) -> dict[str, Any]:
        """The Anthropic tool-definition shape sent in the `tools` array."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
