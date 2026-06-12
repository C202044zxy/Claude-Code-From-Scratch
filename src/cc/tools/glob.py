"""Glob tool — find files by name pattern, newest first."""

from __future__ import annotations

import glob as _glob
import os

from .base import Tool

MAX_RESULTS = 200


class GlobTool(Tool):
    name = "glob"
    description = (
        "Find files matching a glob pattern (e.g. '**/*.py', 'src/**/test_*.py'). "
        "Returns matching paths sorted by most recently modified first."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern to match."},
            "path": {
                "type": "string",
                "description": "Directory to search from (default: current directory).",
            },
        },
        "required": ["pattern"],
    }

    def run(self, pattern: str, path: str = ".") -> str:
        root = path or "."
        full = os.path.join(root, pattern)
        matches = [m for m in _glob.glob(full, recursive=True) if os.path.isfile(m)]
        if not matches:
            return "[no files matched]"
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        truncated = matches[:MAX_RESULTS]
        out = "\n".join(truncated)
        if len(matches) > MAX_RESULTS:
            out += f"\n[+{len(matches) - MAX_RESULTS} more truncated]"
        return out
