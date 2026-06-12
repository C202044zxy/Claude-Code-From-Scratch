"""SWE-bench runner — apply the agent to a task instance, capture its patch.

This is the first system layered onto the loop for the benchmark (see the
roadmap in `README.md`). It does one job: turn a SWE-bench instance into a
*prediction* the official harness can score. For each instance it

1. checks out the instance's repo at `base_commit`,
2. runs `Agent.run(problem_statement)` with the repo as the working directory,
3. captures the resulting `git diff` as the predicted patch, and
4. writes one JSON line per instance to a predictions file.

It does **not** score. Scoring is the official harness's job (`pip install
swebench`), which builds the per-instance environment, applies the predicted
patch plus the gold test patch, and runs the tests. After this runner writes
`predictions.jsonl`:

    python -m swebench.harness.run_evaluation \\
        --dataset_name princeton-nlp/SWE-bench_Lite \\
        --predictions_path predictions.jsonl \\
        --run_id cc-run-1

Usage:
    uv run cc-swebench --dataset princeton-nlp/SWE-bench_Lite --limit 1
    uv run cc-swebench --instances django__django-11099 sympy__sympy-20154
    uv run cc-swebench --dataset ./instances.jsonl --output preds.jsonl

Loading instances from the Hugging Face Hub needs the optional `datasets`
dependency (`uv sync --group bench`). A local `.json`/`.jsonl` path is read
with the standard library, no extra dependency required.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Iterable

from .agent import Agent

# Per-million-token prices for the cost estimate (USD). Keyed by model id.
# Source: the Anthropic pricing the project targets (Opus 4.8 = $5 in / $25 out).
# Cache reads bill at ~0.1x input; this loop sends no cache_control, so the
# cache fields are 0 and only the in/out rates apply.
PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-8": {"input": 5.0, "output": 25.0, "cache_read": 0.5},
}

# What we ask the agent to do with each instance. The problem statement (the
# GitHub issue text) is appended after this framing.
TASK_TEMPLATE = """\
You are resolving a real GitHub issue in the repository checked out at the \
current working directory. The repository is already at the correct commit.

Resolve the issue described below by editing the project's source code. Do not \
write new tests or edit existing test files — the issue will be graded against \
a hidden test suite. Investigate the relevant code, make the smallest change \
that fixes the issue, and verify your reasoning.

--- ISSUE ---
{problem_statement}
"""


def load_instances(
    dataset: str, split: str, instance_ids: list[str] | None
) -> list[dict[str, Any]]:
    """Load SWE-bench instances from a local file or the Hugging Face Hub.

    A path ending in .json/.jsonl is read directly (no extra dependency).
    Anything else is treated as a Hub dataset name and loaded via `datasets`.
    `instance_ids`, if given, filters the result (and preserves its order).
    """
    if dataset.endswith((".json", ".jsonl")):
        instances = _load_local(dataset)
    else:
        instances = _load_hub(dataset, split)

    if instance_ids:
        by_id = {i["instance_id"]: i for i in instances}
        missing = [iid for iid in instance_ids if iid not in by_id]
        if missing:
            raise SystemExit(f"instance ids not found in dataset: {missing}")
        return [by_id[iid] for iid in instance_ids]
    return instances


def _load_local(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    return data if isinstance(data, list) else data.get("instances", [])


def _load_hub(dataset: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "loading from the Hugging Face Hub needs the 'datasets' package. "
            "Install it with `uv sync --group bench`, or pass a local "
            ".json/.jsonl path instead."
        )
    return list(load_dataset(dataset, split=split))


def prepare_repo(instance: dict[str, Any], repos_dir: str, emit) -> str:
    """Clone (once, then cache) the instance's repo and check out base_commit.

    Returns the absolute path to the working copy. The clone is shared across
    instances of the same repo and reset to a pristine base_commit each time.
    """
    repo = instance["repo"]  # e.g. "django/django"
    base_commit = instance["base_commit"]
    repo_path = os.path.join(repos_dir, repo.replace("/", "__"))

    if not os.path.isdir(repo_path):
        emit(f"  cloning {repo} …")
        _git(["clone", f"https://github.com/{repo}.git", repo_path], cwd=repos_dir)

    # Reset to a pristine tree, then fetch+checkout the base commit. fetch is a
    # no-op if we already have it (shallow clones may not, hence the fallback).
    _git(["reset", "--hard"], cwd=repo_path)
    _git(["clean", "-fdx"], cwd=repo_path)
    if _git(["checkout", "-f", base_commit], cwd=repo_path, check=False) != 0:
        _git(["fetch", "--all"], cwd=repo_path)
        _git(["checkout", "-f", base_commit], cwd=repo_path)
    return repo_path


def capture_patch(repo_path: str) -> str:
    """Return the agent's changes as a git diff against base_commit (HEAD).

    Stages everything first so newly created files are included, then diffs the
    index against HEAD (which is the detached base_commit).
    """
    _git(["add", "-A"], cwd=repo_path)
    proc = subprocess.run(
        ["git", "diff", "--cached", "--no-color"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def run_instance(
    instance: dict[str, Any],
    repos_dir: str,
    model: str | None,
    effort: str | None,
    provider: str | None,
    max_turns: int,
    emit,
) -> dict[str, Any]:
    """Run the agent on one instance and return a prediction record."""
    iid = instance["instance_id"]
    emit(f"\n\033[1m▶ {iid}\033[0m ({instance['repo']})")

    repo_path = prepare_repo(instance, repos_dir, emit)
    cwd = os.getcwd()
    os.chdir(repo_path)
    start = time.time()
    try:
        agent = Agent(
            model=model,
            effort=effort,
            provider=provider,
            max_turns=max_turns,
            emit=emit,
        )
        task = TASK_TEMPLATE.format(problem_statement=instance["problem_statement"])
        try:
            agent.run(task)
        except Exception as e:  # noqa: BLE001 — one bad instance shouldn't kill the run
            emit(f"  \033[31magent error: {type(e).__name__}: {e}\033[0m")
        patch = capture_patch(repo_path)
    finally:
        os.chdir(cwd)

    elapsed = time.time() - start
    cost = estimate_cost(agent.model, agent.usage)
    emit(
        f"  done in {elapsed:.0f}s · {agent.turns} turns · "
        f"{agent.usage['input_tokens']:,} in / {agent.usage['output_tokens']:,} out"
        f" · ~${cost:.2f} · patch {len(patch)} chars"
    )

    return {
        "instance_id": iid,
        "model_name_or_path": agent.model,
        "model_patch": patch,
        # Extra fields (ignored by the harness) for our own analysis.
        "cc_turns": agent.turns,
        "cc_usage": agent.usage,
        "cc_cost_usd": round(cost, 4),
        "cc_seconds": round(elapsed, 1),
    }


def estimate_cost(model: str, usage: dict[str, int]) -> float:
    """USD cost of one instance from its token usage and the model's prices."""
    price = PRICING.get(model)
    if price is None:
        return 0.0
    return (
        usage["input_tokens"] / 1e6 * price["input"]
        + usage["output_tokens"] / 1e6 * price["output"]
        + usage["cache_read_input_tokens"] / 1e6 * price.get("cache_read", 0.0)
    )


def _git(args: list[str], cwd: str, check: bool = True) -> int:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed in {cwd}:\n{proc.stderr}")
    return proc.returncode


def _print_summary(results: Iterable[dict[str, Any]]) -> None:
    results = list(results)
    if not results:
        return
    total_cost = sum(r["cc_cost_usd"] for r in results)
    total_in = sum(r["cc_usage"]["input_tokens"] for r in results)
    total_out = sum(r["cc_usage"]["output_tokens"] for r in results)
    n = len(results)
    print("\n" + "─" * 60)
    print(f"{n} instance(s) · ${total_cost:.2f} total · ${total_cost / n:.2f}/instance")
    print(
        f"avg {total_in // n:,} input + {total_out // n:,} output tokens per instance"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="cc-swebench",
        description="Run the agent on SWE-bench instances and write predictions.",
    )
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Lite",
        help="Hub dataset name or path to a local .json/.jsonl (default: SWE-bench_Lite).",
    )
    parser.add_argument("--split", default="test", help="Dataset split (default: test).")
    parser.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Specific instance_ids to run (default: all in the dataset).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Run only the first N instances."
    )
    parser.add_argument(
        "--output",
        default="predictions.jsonl",
        help="Where to write predictions (default: predictions.jsonl).",
    )
    parser.add_argument(
        "--repos-dir",
        default=".swebench-repos",
        help="Directory for cached repo clones (default: .swebench-repos).",
    )
    parser.add_argument("--model", default=None, help="Override the model id.")
    parser.add_argument("--provider", default=None, help="LLM provider.")
    parser.add_argument("--effort", default=None, help="Reasoning effort.")
    parser.add_argument(
        "--max-turns", type=int, default=60, help="Max agent turns per instance."
    )
    args = parser.parse_args()

    # Load .env so ANTHROPIC_API_KEY is available, matching the cc CLI.
    from .cli import _load_dotenv

    _load_dotenv()

    instances = load_instances(args.dataset, args.split, args.instances)
    if args.limit is not None:
        instances = instances[: args.limit]
    if not instances:
        print("no instances to run", file=sys.stderr)
        return 1

    repos_dir = os.path.abspath(args.repos_dir)
    os.makedirs(repos_dir, exist_ok=True)
    output = os.path.abspath(args.output)

    print(f"running {len(instances)} instance(s) → {output}")
    results = []
    with open(output, "w") as f:
        for instance in instances:
            record = run_instance(
                instance,
                repos_dir,
                args.model,
                args.effort,
                args.provider,
                args.max_turns,
                emit=print,
            )
            f.write(json.dumps(record) + "\n")
            f.flush()
            results.append(record)

    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
