"""Bash tool — the single tool that gives the agent the most surface area.

Almost any action (run tests, git, install deps, inspect the system) is reachable
through a shell. We promote a few read/edit actions to dedicated tools (read/edit/
glob/grep) so the harness can render and reason about them, but bash is the
workhorse.
"""

from __future__ import annotations

import os
import subprocess

from .base import Tool, ToolError


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a bash command in the current working directory and return its "
        "combined stdout/stderr. Use for running tests, git, building, installing "
        "dependencies, or any shell action. Commands run with a timeout."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to run.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120, max 600).",
            },
        },
        "required": ["command"],
    }

    def run(self, command: str, timeout: int = 120) -> str:
        timeout = min(max(int(timeout), 1), 600)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"Command timed out after {timeout}s: {command}")

        out = proc.stdout
        if proc.stderr:
            out += ("\n" if out else "") + proc.stderr
        if proc.returncode != 0:
            out += f"\n[exit code: {proc.returncode}]"
        return out.strip() or "[no output]"
