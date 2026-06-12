"""Command-line entry point.

Usage:
    cc "fix the failing test in tests/test_foo.py"
    cc --workdir /path/to/repo "add a --verbose flag to the CLI"
    echo "your task" | cc            # task from stdin

The CLI's only jobs: pick the working directory the agent operates in, read the
task, run the agent loop, and print the final answer.
"""

from __future__ import annotations

import argparse
import os
import sys

from .agent import Agent


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="cc",
        description="A minimal Claude Code, built from first principles.",
    )
    parser.add_argument("task", nargs="*", help="The task for the agent.")
    parser.add_argument(
        "-C", "--workdir", default=".", help="Directory to run in (default: cwd)."
    )
    parser.add_argument("--model", default=None, help="Override the model id.")
    parser.add_argument(
        "--effort",
        default=None,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort (default: high).",
    )
    args = parser.parse_args()

    task = " ".join(args.task).strip()
    if not task and not sys.stdin.isatty():
        task = sys.stdin.read().strip()
    if not task:
        parser.error("no task provided (pass it as an argument or via stdin)")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "error: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and "
            "fill it in, or export the key.",
            file=sys.stderr,
        )
        return 1

    workdir = os.path.abspath(args.workdir)
    if not os.path.isdir(workdir):
        print(f"error: not a directory: {workdir}", file=sys.stderr)
        return 1
    os.chdir(workdir)

    print(f"\033[1mcc\033[0m in {workdir}\n\033[2m> {task}\033[0m\n")

    agent = Agent(model=args.model, effort=args.effort)
    try:
        final = agent.run(task)
    except KeyboardInterrupt:
        print("\n[interrupted]", file=sys.stderr)
        return 130

    print("\n" + "─" * 60)
    print(final or "[no final message]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
