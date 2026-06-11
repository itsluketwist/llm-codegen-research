"""Class to access LLMs via the TogetherAI API."""

import re
from typing import Any, cast

import together

from llm_cgr.llm.clients.base import Base_LLM


# matches a <think>...</think> block at the start of a response, used by
# models that embed their reasoning trace directly in the content
_THINK_BLOCK = re.compile(r"\A<think>(.*?)</think>\s*", re.DOTALL)


class TogetherAI_LLM(Base_LLM):
    """Class to access LLMs via the TogetherAI API."""

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
        Initialise the TogetherAI client.

        Requires the TOGETHER_API_KEY environment variable to be set.
        Set enable_reasoning=True when using a reasoning model (e.g. deepseek-ai/DeepSeek-R1).
        """
        super().__init__(
            model=model,
            system=system,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_reasoning=enable_reasoning,
        )
        self._client = together.Together()

    def _build_message(
        self,
        role: str,
        content: str,
    ) -> dict[str, str]:
        """Build a TogetherAI model message."""
        return {"role": role, "content": content}

    def _build_input(
        self,
        user: str,
        system: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the full TogetherAI model input."""
        input = []
        if system:
            input.append(self._build_message(role="system", content=system))
        input.append(self._build_message(role="user", content=user))
        return input

    def _get_response(
        self,
        model: str,
        input: list[dict[str, Any]],
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, str | None]:
        """Generate a model response from the TogetherAI API."""
        response = self._client.chat.completions.create(
            model=model,
            messages=cast(Any, input),
            temperature=temperature if temperature is not None else together.omit,
            top_p=top_p if top_p is not None else together.omit,
            max_tokens=max_tokens if max_tokens is not None else together.omit,
        )
        # cast to Any first as together doesn't publicly export the message type,
        # then cast content to str as text completions always have it set
        message = cast(Any, response.choices[0].message)
        content = cast(str, message.content)

        # chain-of-thought from reasoning models: most (e.g. DeepSeek-R1) use
        # reasoning_content, some (e.g. Kimi-K2.6) use reasoning; None otherwise
        reasoning = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        )

        # some models embed their reasoning as a <think>...</think> block at
        # the start of content instead of a separate field; pull it out
        if reasoning is None:
            think_match = _THINK_BLOCK.match(content)
            if think_match:
                reasoning = think_match.group(1).strip()
                content = content[think_match.end() :]

        return content, reasoning
