"""Write tool — create or overwrite a file with new contents."""

from __future__ import annotations

import os

from .base import Tool, ToolError


class WriteTool(Tool):
    name = "write"
    description = (
        "Write content to a file, creating it (and any parent directories) or "
        "overwriting it entirely. For small edits to an existing file, prefer the "
        "`edit` tool instead."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write."},
            "content": {"type": "string", "description": "The full file content."},
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str) -> str:
        parent = os.path.dirname(os.path.abspath(path))
        try:
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            raise ToolError(f"Could not write {path}: {e}")
        n = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        return f"Wrote {n} lines to {path}"
