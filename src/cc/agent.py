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

import anthropic

from .prompts import SYSTEM_PROMPT
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
        # Anthropic-only knobs the loop passes; DeepSeek rejects these.
        "extended": True,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/anthropic",
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "default_max_tokens": 8000,
        "extended": False,
    },
}


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

    def run(self, task: str) -> str:
        """Run the agent to completion on `task`. Returns the final text answer."""
        self.messages.append({"role": "user", "content": task})

        for _turn in range(self.max_turns):
            # thinking/output_config are Anthropic-only; providers like DeepSeek
            # speak the same API but reject them, so only send them when extended.
            extra: dict[str, Any] = {}
            if self.extended:
                extra["thinking"] = {"type": "adaptive"}
                extra["output_config"] = {"effort": self.effort}

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system,
                tools=[t.schema() for t in self.tools],
                messages=self.messages,
                **extra,
            )

            # Append the assistant turn verbatim — this preserves thinking blocks
            # and tool_use blocks the API needs to see on the next request.
            self.messages.append({"role": "assistant", "content": response.content})
            self._render_assistant(response.content)

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

    def _render_assistant(self, content: list[Any]) -> None:
        for block in content:
            if block.type == "text" and block.text.strip():
                self.emit(block.text.strip())

    @staticmethod
    def _final_text(content: list[Any]) -> str:
        parts = [b.text for b in content if b.type == "text"]
        return "\n".join(parts).strip()


def _fmt_args(args: dict[str, Any]) -> str:
    """Compact one-line arg preview for the progress log."""
    out = json.dumps(args)
    return out if len(out) <= 120 else out[:117] + "…"
