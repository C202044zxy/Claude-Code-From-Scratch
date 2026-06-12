"""Edit tool — exact string replacement in an existing file.

The invariant that makes this safe: `old_string` must match exactly once (unless
replace_all). If it matches zero times, the model's mental model of the file is
stale and we reject rather than guess. If it matches many times, the edit is
ambiguous. This forces the model to read first and target unique context.
"""

from __future__ import annotations

import os

from .base import Tool, ToolError


class EditTool(Tool):
    name = "edit"
    description = (
        "Replace an exact string in a file with a new string. `old_string` must "
        "appear exactly once in the file (include surrounding context to make it "
        "unique), unless `replace_all` is true. Read the file first so your "
        "old_string matches the current contents."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit."},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of requiring uniqueness.",
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    def run(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        if not os.path.exists(path):
            raise ToolError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        count = content.count(old_string)
        if count == 0:
            raise ToolError(
                "old_string not found in file. Read the file again — its contents "
                "may differ from what you expected."
            )
        if count > 1 and not replace_all:
            raise ToolError(
                f"old_string is ambiguous: found {count} matches. Add surrounding "
                "context to make it unique, or set replace_all=true."
            )

        content = content.replace(old_string, new_string)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        where = f"{count} occurrences" if replace_all else "1 occurrence"
        return f"Edited {path} ({where} replaced)"
