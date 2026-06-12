"""Read tool — read a file with line numbers (cat -n style).

Line numbers matter: they give the model stable coordinates to talk about and let
the edit tool target unique strings. We cap the output so a giant file can't blow
the context window in one call.
"""

from __future__ import annotations

import os

from .base import Tool, ToolError

MAX_LINES = 2000
MAX_LINE_LEN = 2000


class ReadTool(Tool):
    name = "read"
    description = (
        "Read a text file from the local filesystem. Returns the contents with a "
        "line number prefix on each line. Use an absolute path, or a path relative "
        "to the current working directory."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
            "offset": {
                "type": "integer",
                "description": "1-based line number to start reading from (optional).",
            },
            "limit": {
                "type": "integer",
                "description": f"Max number of lines to read (default {MAX_LINES}).",
            },
        },
        "required": ["path"],
    }

    def run(self, path: str, offset: int = 1, limit: int = MAX_LINES) -> str:
        if not os.path.exists(path):
            raise ToolError(f"File not found: {path}")
        if os.path.isdir(path):
            raise ToolError(f"Path is a directory, not a file: {path}")

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as e:
            raise ToolError(f"Could not read {path}: {e}")

        offset = max(int(offset), 1)
        limit = min(int(limit), MAX_LINES)
        window = lines[offset - 1 : offset - 1 + limit]
        if not window:
            return "[file is empty or offset is past end of file]"

        rendered = []
        for i, line in enumerate(window, start=offset):
            text = line.rstrip("\n")
            if len(text) > MAX_LINE_LEN:
                text = text[:MAX_LINE_LEN] + "… [truncated]"
            rendered.append(f"{i:6d}\t{text}")
        return "\n".join(rendered)
