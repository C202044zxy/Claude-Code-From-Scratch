"""Tool registry.

`default_tools()` returns the irreducible set that gives the agent enough surface
area to solve real coding tasks: a shell plus file read/write/edit and search.
Add a capability by appending a Tool here.
"""

from __future__ import annotations

from .base import Tool, ToolError
from .bash import BashTool
from .edit import EditTool
from .glob import GlobTool
from .grep import GrepTool
from .read import ReadTool
from .write import WriteTool


def default_tools() -> list[Tool]:
    return [
        BashTool(),
        ReadTool(),
        WriteTool(),
        EditTool(),
        GlobTool(),
        GrepTool(),
    ]


__all__ = ["Tool", "ToolError", "default_tools"]
