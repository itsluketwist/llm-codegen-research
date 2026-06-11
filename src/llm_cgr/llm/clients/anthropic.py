"""Classes to access LLMs via the Anthropic Claude API."""

from typing import Any, cast

import anthropic
from anthropic.types import (
    MessageParam,
    TextBlock,
    ThinkingBlock,
    ThinkingConfigEnabledParam,
)

from llm_cgr.defaults import DEFAULT_MAX_TOKENS, DEFAULT_THINKING_BUDGET
from llm_cgr.llm.clients.base import Base_LLM


class Anthropic_LLM(Base_LLM):
    """Class to access LLMs via the Anthropic API."""

    def __init__(
        self,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        enable_reasoning: bool = False,
    ) -> None:
        """
        Initialise the Anthropic client.

        Requires the ANTHROPIC_API_KEY environment variable to be set.
        Set enable_reasoning=True to enable extended thinking on supported models
        (e.g. claude-sonnet-4-5).
        """
        super().__init__(
            model=model,
            system=system,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_reasoning=enable_reasoning,
        )
        self._client = anthropic.Anthropic()

    def _build_message(
        self,
        role: str,
        content: str,
    ) -> dict[str, str | list[dict[str, str]]]:
        """Build an Anthropic model message."""
        return {
            "role": role,
            "content": [
                {
                    "type": "text",
                    "text": content,
                }
            ],
        }

    def _build_input(
        self,
        user: str,
        system: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the full Anthropic model input."""
        return [self._build_message(role="user", content=user)]

    def _get_response(
        self,
        model: str,
        input: list[dict[str, Any]],
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, str | None]:
        """Generate a model response from the Anthropic API."""
        # extended thinking is incompatible with custom temperature/top_p
        thinking = (
            ThinkingConfigEnabledParam(
                type="enabled",
                budget_tokens=DEFAULT_THINKING_BUDGET,
            )
            if self._enable_reasoning
            else anthropic.omit
        )
        # custom temperature/top_p are not supported alongside extended thinking,
        # and the api rejects requests that set both temperature and top_p
        _temperature = (
            temperature
            if temperature is not None and not self._enable_reasoning
            else anthropic.omit
        )
        _top_p = (
            top_p
            if top_p is not None
            and not self._enable_reasoning
            and _temperature is anthropic.omit
            else anthropic.omit
        )

        response = self._client.messages.create(
            model=model,
            system=system or self._system or anthropic.omit,
            messages=cast(list[MessageParam], input),
            temperature=_temperature,
            top_p=_top_p,
            max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
            thinking=thinking,
        )

        # collect chain-of-thought from any thinking blocks; None if not present
        thinking_blocks = [
            block.thinking
            for block in response.content
            if isinstance(block, ThinkingBlock)
        ]
        reasoning = "\n".join(thinking_blocks) if thinking_blocks else None

        # the final answer is always returned as a text block
        text_block = next(
            block for block in response.content if isinstance(block, TextBlock)
        )
        return text_block.text, reasoning
