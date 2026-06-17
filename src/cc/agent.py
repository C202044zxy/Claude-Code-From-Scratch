"""The agent loop — the ~50-line heart of the whole system.

The shape, straight from the Anthropic Messages API tool-use pattern:

    loop:
        response = model(messages, tools, system)
        if response wants tools:
            execute each tool, append results
            continue
        else:
            done

Everything else in this project (the tools, the prompt, the CLI) exists to feed
this loop. "The model controls the loop": we never decide what to do next — we
just run what the model asks for and hand back the results.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable
import sys

import anthropic

from .prompts import SYSTEM_PROMPT, COMPACTION_PROMPT
from .tools import Tool, ToolError, default_tools

# A sink for human-readable progress. The CLI passes `print`; tests can capture.
Emit = Callable[[str], None]

# Provider presets. Both speak the Anthropic Messages API — DeepSeek ships an
# Anthropic-compatible endpoint — so the SDK, the message format, and the tool
# format are identical. Only the base_url, the API key, and a couple of
# Anthropic-only request params differ. To add another such provider, add a row.
PROVIDERS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "base_url": None,  # SDK default
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-opus-4-8",
        "default_max_tokens": 16000,
        # Fix(Claude): __init__ reads cfg["context_window"] for the compaction
        # budget, but the provider rows never defined it (KeyError on construct).
        "context_window": 200_000,
        # Anthropic-only knobs the loop passes; DeepSeek rejects these.
        "extended": True,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/anthropic",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-v4-flash",
        "default_max_tokens": 16000,  # v4-flash caps output at 384K; this is a sane default
        "context_window": 1_000_000,
        "extended": False,
    },
}


def _stdout_write(text: str):
    sys.stdout.write(text)
    sys.stdout.flush()


class Agent:
    def __init__(
        self,
        tools: list[Tool] | None = None,
        system: str = SYSTEM_PROMPT,
        model: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
        provider: str | None = None,
        emit: Emit = print,
        max_turns: int = 60,
        compaction_prompt: str = COMPACTION_PROMPT,
        context_window: int | None = None,
        compact_threshold: float | None = None,
        keep_recent: int | None = None,
        stream_to: Emit = _stdout_write,
    ) -> None:
        provider = (provider or os.getenv("CC_PROVIDER", "anthropic")).lower()
        if provider not in PROVIDERS:
            raise ValueError(
                f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}"
            )
        self.provider = provider
        cfg = PROVIDERS[provider]
        self.extended = cfg["extended"]

        # base_url=None lets the SDK use its default; api_key=None lets it fall
        # back to the env var it reads natively (ANTHROPIC_API_KEY).
        self.client = anthropic.Anthropic(
            base_url=cfg["base_url"],
            api_key=os.getenv(cfg["api_key_env"]),
        )
        self.tools = tools if tools is not None else default_tools()
        self.tools_by_name = {t.name: t for t in self.tools}
        self.system = system
        self.model = model or os.getenv("CC_MODEL") or cfg["default_model"]
        self.max_tokens = max_tokens or int(
            os.getenv("CC_MAX_TOKENS") or cfg["default_max_tokens"]
        )
        self.effort = effort or os.getenv("CC_EFFORT", "high")
        self.emit = emit
        self.max_turns = max_turns
        self.messages: list[dict[str, Any]] = []
        # Running token totals across every API call this Agent makes. Useful for
        # cost accounting (see swebench.py). Cache fields stay 0 unless the
        # provider/request enables prompt caching.
        self.turns = 0
        self.usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

        self.compaction_prompt = compaction_prompt
        self.context_window = context_window or int(
            os.getenv("CC_CONTEXT_WINDOW") or cfg["context_window"]
        )
        self.compact_threshold = compact_threshold or float(
            os.getenv("CC_COMPACT_THRESHOLD", 0.8)
        )
        self.keep_recent = keep_recent or int(os.getenv("CC_KEEP_RECENT", 6))
        self.last_context_tokens = 0
        self.compactions = 0

        self.stream_to = stream_to

    def run(self, task: str) -> str:
        """Run the agent to completion on `task`. Returns the final text answer."""
        self.messages.append({"role": "user", "content": task})
        self.task = task

        for _turn in range(self.max_turns):
            self._maybe_compact()

            # thinking/output_config are Anthropic-only; providers like DeepSeek
            # speak the same API but reject them, so only send them when extended.
            extra: dict[str, Any] = {}
            if self.extended:
                extra["thinking"] = {"type": "adaptive"}
                extra["output_config"] = {"effort": self.effort}

            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system,
                tools=[t.schema() for t in self.tools],
                messages=self.messages,
                **extra,
            ) as stream:
                self._render_stream(stream)
                response = stream.get_final_message()

            self.turns += 1
            self._tally_usage(response)
            self.last_context_tokens = _context_tokens(response)

            # Append the assistant turn verbatim — this preserves thinking blocks
            # and tool_use blocks the API needs to see on the next request.
            self.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return self._final_text(response.content)

            # Execute every requested tool and return all results in one user turn.
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result, is_error = self._execute(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                        "is_error": is_error,
                    }
                )
            self.messages.append({"role": "user", "content": tool_results})

        return "[stopped: hit max turns without finishing]"

    def _tally_usage(self, response: Any) -> None:
        """Add one response's token counts to the running totals."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        for key in self.usage:
            self.usage[key] += getattr(usage, key, 0) or 0

    def _execute(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        tool = self.tools_by_name.get(name)
        if tool is None:
            return f"Unknown tool: {name}", True
        self.emit(f"  \033[2m· {name}({_fmt_args(args)})\033[0m")
        try:
            return tool.run(**args), False
        except ToolError as e:
            return str(e), True
        except Exception as e:  # noqa: BLE001 — surface, don't crash the loop
            return f"{type(e).__name__}: {e}", True

    ## ----- Compaction -----
    def _should_compact(self) -> bool:
        if self.last_context_tokens <= self.compact_threshold * self.context_window:
            return False
        if len(self.messages) <= self.keep_recent + 2:
            return False
        return True

    def _compact(self) -> None:
        cut = len(self.messages) - self.keep_recent
        while cut > 0 and self.messages[cut]["role"] != "assistant":
            cut -= 1
        if cut == 0:
            return

        prefix, tail = self.messages[:cut], self.messages[cut:]
        transcript = self._render_transcript(prefix)
        summary = self._summarize(transcript)
        self.messages = self._rebuild_after_compaction(self.task, summary, tail)
        self.compactions += 1
        self.last_context_tokens = 0
        self.emit(
            f"  \033[2m· compacted {len(prefix)} msgs → summary\033[0m"
        )

    def _maybe_compact(self) -> None:
        if self._should_compact():
            self._compact()

    def _render_transcript(self, messages: list[dict[str, Any]]) -> str:
        """Flatten a prefix of `messages` into plain text for the summarizer.

        Pure: no API calls, no mutation. Renders text blocks, tool calls
        (name + args) and tool results; skips `thinking` blocks; truncates long
        blocks. Plain text sidesteps the API's tool_use/tool_result pairing
        rules entirely, so the summary call never sees an orphaned block.
        """
        lines: list[str] = []
        for msg in messages:
            role = str(msg["role"]).upper()
            content = msg["content"]
            # The initial task and a post-compaction summary are plain strings.
            if isinstance(content, str):
                lines.append(f"{role}: {_truncate(content)}")
                continue
            for block in content:
                rendered = _render_block(block)
                if rendered:
                    lines.append(f"{role}: {rendered}")
        return "\n".join(lines)

    def _summarize(self, text: str) -> str:
        extra: dict[str, Any] = {}
        if self.extended:
            extra["output_config"] = {"effort": "low"}
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self.compaction_prompt,
            messages=[{"role": "user", "content": text}],
            **extra,
        )
        self._tally_usage(response)
        return self._final_text(response.content)

    def _rebuild_after_compaction(
        self, task: str, summary: str, tail: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": (
                    task
                    + "\n\n[Earlier conversation compacted to save context. "
                    + "Summary of work so far:]\n"
                    + summary
                ),
            },
            *tail,
        ]
    
    # ----- Streaming -----
    def _render_stream(self, stream: Any):
        in_thinking = False
        wrote_any = False

        for event in stream:
            if event.type == "thinking":
                if not in_thinking:
                    in_thinking = True
                    self.stream_to("\033[2m")
                self.stream_to(event.thinking)
                wrote_any = True
            elif event.type == "text":
                if in_thinking:
                    in_thinking = False
                    self.stream_to("\033[0m\n")
                self.stream_to(event.text)
                wrote_any = True

        if in_thinking:
            self.stream_to("\033[0m")
        if wrote_any:
            self.stream_to("\n")

    @staticmethod
    def _final_text(content: list[Any]) -> str:
        parts = [b.text for b in content if b.type == "text"]
        return "\n".join(parts).strip()


def _context_tokens(response: Any) -> int:
    """Size of the prompt the model just saw, from one response's usage.

    Sums the three input-side fields so the reading is correct whether or not
    prompt caching is on (cache_read/cache_creation are 0 when it's off).
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0
    return (
        (getattr(usage, "input_tokens", 0) or 0)
        + (getattr(usage, "cache_read_input_tokens", 0) or 0)
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
    )


def _fmt_args(args: dict[str, Any]) -> str:
    """Compact one-line arg preview for the progress log."""
    out = json.dumps(args)
    return out if len(out) <= 120 else out[:117] + "…"


# Max chars kept from any one block when rendering a transcript; longer blocks
# are truncated so the summary call stays cheap and the text stays readable.
_TRANSCRIPT_BLOCK_LIMIT = 2000


def _truncate(text: str, limit: int = _TRANSCRIPT_BLOCK_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [+{len(text) - limit} chars]"


def _block_field(block: Any, name: str) -> Any:
    """Read a field from a content block that may be an SDK object or a dict.

    Assistant blocks are SDK objects (`block.type`); the tool_result blocks the
    loop builds are plain dicts (`block["type"]`). This reads either.
    """
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def _render_block(block: Any) -> str:
    """Render one content block to text, or "" to skip it (e.g. thinking)."""
    btype = _block_field(block, "type")
    if btype == "text":
        return _truncate((_block_field(block, "text") or "").strip())
    if btype == "tool_use":
        name = _block_field(block, "name")
        args = _fmt_args(_block_field(block, "input") or {})
        return f"[tool_use {name}({args})]"
    if btype == "tool_result":
        content = _block_field(block, "content")
        text = content if isinstance(content, str) else json.dumps(content)
        label = "tool_error" if _block_field(block, "is_error") else "tool_result"
        return f"[{label}: {_truncate(text)}]"
    # thinking / redacted_thinking / anything unknown: nothing useful to summarize.
    return ""
