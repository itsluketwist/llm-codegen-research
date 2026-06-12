"""Class to access LLMs via the Qwen API."""

import os
from typing import Any, cast

import openai
from openai.types.chat import ChatCompletionMessageParam

from llm_cgr.llm.clients.base import Base_LLM


class Qwen_LLM(Base_LLM):
    """Class to access LLMs via the Qwen API, using the OpenAI interfaces."""

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
        Initialise the Qwen client.

        Requires the QWEN_API_KEY environment variable to be set.
        Set enable_reasoning=True to enable thinking mode on supported models
        (e.g. qwen3-235b-a22b); thinking is off by default.
        """
        super().__init__(
            model=model,
            system=system,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_reasoning=enable_reasoning,
        )
        self._client = openai.OpenAI(
            api_key=os.environ["QWEN_API_KEY"],
            base_url=os.environ["QWEN_API_URL"],
        )

    def _build_message(
        self,
        role: str,
        content: str,
    ) -> dict[str, str]:
        """Build an OpenAI model message."""
        return {"role": role, "content": content}

    def _build_input(
        self,
        user: str,
        system: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the full OpenAI model input."""
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
        """Generate a model response from the Qwen API."""
        response = self._client.chat.completions.create(
            messages=cast(list[ChatCompletionMessageParam], input),
            model=model,
            temperature=temperature if temperature is not None else openai.omit,
            top_p=top_p if top_p is not None else openai.omit,
            max_completion_tokens=max_tokens if max_tokens is not None else openai.omit,
            # thinking mode and other dashscope extras (e.g. web search) are left
            # off unless reasoning is explicitly requested
            extra_body={"enable_thinking": True} if self._enable_reasoning else None,
        )
        message = response.choices[0].message

        # chain-of-thought from thinking-enabled models; None otherwise
        reasoning = getattr(message, "reasoning_content", None)

        # cast to str as text completions always return string content
        return cast(str, message.content), reasoning
