"""Generic OpenAI-compatible LLM client that can point at any endpoint."""

import os
from typing import Any, cast

import openai
from openai.types.chat import ChatCompletionMessageParam

from llm_cgr.llm.clients.base import Base_LLM


class Custom_LLM(Base_LLM):
    """Generic client for any OpenAI-compatible API endpoint.

    Rather than hardcoding a provider URL and API key, the caller specifies
    which environment variables hold those values, making it trivial to switch
    between any OpenAI-compatible provider without code changes.
    """

    def __init__(
        self,
        api_key_env: str = "LLM_CGR_API_KEY",
        base_url_env: str = "LLM_CGR_BASE_URL",
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        enable_reasoning: bool = False,
    ) -> None:
        """
        Initialise a custom OpenAI-compatible client.

        api_key_env is the name of the environment variable holding the API key,
        and base_url_env is the name of the environment variable holding the
        endpoint base URL (e.g. 'https://my-provider.com/v1').
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
            api_key=os.environ[api_key_env],
            base_url=os.environ[base_url_env],
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
        """Generate a model response from the configured endpoint."""
        response = self._client.chat.completions.create(
            messages=cast(list[ChatCompletionMessageParam], input),
            model=model,
            temperature=temperature if temperature is not None else openai.omit,
            top_p=top_p if top_p is not None else openai.omit,
            max_completion_tokens=max_tokens if max_tokens is not None else openai.omit,
        )
        message = response.choices[0].message

        # reasoning_content is returned by some endpoints (e.g. deepseek-style);
        # gracefully return None if this endpoint does not support it
        reasoning = getattr(message, "reasoning_content", None)

        # cast to str as text completions always return string content
        return cast(str, message.content), reasoning
