"""Class to access LLMs via the MistralAI API."""

import os
from typing import Any

from mistralai import client
from mistralai.client.models import TextChunk, ThinkChunk

from llm_cgr.llm.clients.base import Base_LLM


class Mistral_LLM(Base_LLM):
    """Class to access LLMs via the MistralAI API."""

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
        Initialise the Mistral client.

        Requires the MISTRAL_API_KEY environment variable to be set.
        Set enable_reasoning=True to request chain-of-thought from reasoning
        models (e.g. magistral-medium-latest).
        """
        super().__init__(
            model=model,
            system=system,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_reasoning=enable_reasoning,
        )
        self._client = client.Mistral(
            api_key=os.environ["MISTRAL_API_KEY"],
        )

    def _build_message(
        self,
        role: str,
        content: str,
    ) -> dict[str, str | list[dict[str, str]]]:
        """Build a Mistral model message."""
        return {
            "role": role,
            "content": content,
        }

    def _build_input(
        self,
        user: str,
        system: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the full Mistral model input."""
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
        """Generate a model response from the MistralAI API."""
        response = self._client.chat.complete(
            model=model,
            messages=input,
            temperature=temperature if temperature is not None else client.UNSET,
            top_p=top_p,
            max_tokens=max_tokens if max_tokens is not None else client.UNSET,
            reasoning_effort="high" if self._enable_reasoning else client.UNSET,
        )
        content = response.choices[0].message.content

        # plain string content means no reasoning chunks were returned
        if isinstance(content, str):
            return content, None

        # otherwise content is a list of chunks: thinking and final text
        reasoning_parts = [
            inner.text
            for chunk in content
            if isinstance(chunk, ThinkChunk)
            for inner in chunk.thinking
            if isinstance(inner, TextChunk)
        ]
        text_parts = [chunk.text for chunk in content if isinstance(chunk, TextChunk)]

        reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
        return "\n".join(text_parts), reasoning
