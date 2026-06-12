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

from .agent import PROVIDERS, Agent


def _load_dotenv(path: str = ".env") -> None:
    """Tiny .env loader (no dependency). Real env vars win over the file."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


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
        "--provider",
        default=None,
        choices=sorted(PROVIDERS),
        help="LLM provider (default: $CC_PROVIDER or anthropic).",
    )
    parser.add_argument(
        "--effort",
        default=None,
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Reasoning effort (default: high). Anthropic only.",
    )
    args = parser.parse_args()

    # Load .env from the launch dir before we chdir into the workdir below.
    _load_dotenv()

    task = " ".join(args.task).strip()
    if not task and not sys.stdin.isatty():
        task = sys.stdin.read().strip()
    if not task:
        parser.error("no task provided (pass it as an argument or via stdin)")

    provider = (args.provider or os.getenv("CC_PROVIDER", "anthropic")).lower()
    key_env = PROVIDERS.get(provider, {}).get("api_key_env", "ANTHROPIC_API_KEY")
    if not os.getenv(key_env):
        print(
            f"error: {key_env} is not set (provider: {provider}). Copy "
            ".env.example to .env and fill it in, or export the key.",
            file=sys.stderr,
        )
        return 1

    workdir = os.path.abspath(args.workdir)
    if not os.path.isdir(workdir):
        print(f"error: not a directory: {workdir}", file=sys.stderr)
        return 1
    os.chdir(workdir)

    print(f"\033[1mcc\033[0m in {workdir}\n\033[2m> {task}\033[0m\n")

    agent = Agent(model=args.model, effort=args.effort, provider=provider)
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
