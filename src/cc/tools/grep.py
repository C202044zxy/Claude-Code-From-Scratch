"""Grep tool — search file contents by regex.

Uses ripgrep (`rg`) when available for speed, and falls back to a pure-Python
walk so the agent works on any machine.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

from .base import Tool, ToolError

MAX_MATCHES = 200


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents for a regular expression. Returns matching lines as "
        "`path:line_number: text`. Optionally restrict to files matching a glob."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex to search for."},
            "path": {
                "type": "string",
                "description": "Directory or file to search (default: current directory).",
            },
            "glob": {
                "type": "string",
                "description": "Only search files matching this glob (e.g. '*.py').",
            },
        },
        "required": ["pattern"],
    }

    def run(self, pattern: str, path: str = ".", glob: str | None = None) -> str:
        if shutil.which("rg"):
            return self._ripgrep(pattern, path, glob)
        return self._python(pattern, path, glob)

    def _ripgrep(self, pattern: str, path: str, glob: str | None) -> str:
        cmd = ["rg", "--line-number", "--no-heading", "--color=never"]
        if glob:
            cmd += ["--glob", glob]
        cmd += [pattern, path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode not in (0, 1):  # 1 = no matches
            raise ToolError(proc.stderr.strip() or "ripgrep failed")
        lines = proc.stdout.splitlines()
        return self._format(lines)

    def _python(self, pattern: str, path: str, glob: str | None) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"Invalid regex: {e}")
        import fnmatch

        results: list[str] = []
        targets: list[str] = []
        if os.path.isfile(path):
            targets = [path]
        else:
            for dirpath, dirnames, filenames in os.walk(path):
                dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".venv"}]
                for fn in filenames:
                    if glob and not fnmatch.fnmatch(fn, glob):
                        continue
                    targets.append(os.path.join(dirpath, fn))

        for fp in targets:
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    for n, line in enumerate(f, 1):
                        if rx.search(line):
                            results.append(f"{fp}:{n}: {line.rstrip()}")
                            if len(results) >= MAX_MATCHES + 1:
                                return self._format(results)
            except OSError:
                continue
        return self._format(results)

    @staticmethod
    def _format(lines: list[str]) -> str:
        if not lines:
            return "[no matches]"
        truncated = lines[:MAX_MATCHES]
        out = "\n".join(truncated)
        if len(lines) > MAX_MATCHES:
            out += f"\n[+{len(lines) - MAX_MATCHES} more matches truncated]"
        return out
