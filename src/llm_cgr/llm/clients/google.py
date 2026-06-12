"""Class to access LLMs via the Google Gemini API."""

import os
from typing import Any, cast

from google import genai
from google.genai import types

from llm_cgr.llm.clients.base import Base_LLM


class Google_LLM(Base_LLM):
    """Class to access LLMs via the Google Gemini API."""

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
        Initialise the Google client.

        Requires the GOOGLE_API_KEY environment variable to be set.
        Set enable_reasoning=True to include the model's thinking summary on
        supported models (e.g. gemini-2.5-pro); thinking output is hidden by
        default.
        """
        super().__init__(
            model=model,
            system=system,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            enable_reasoning=enable_reasoning,
        )
        self._client = genai.Client(
            api_key=os.environ["GOOGLE_API_KEY"],
        )

    def _build_message(
        self,
        role: str,
        content: str,
    ) -> dict[str, Any]:
        """Build a Gemini model message."""
        # gemini uses "model" rather than "assistant" for past responses
        return {
            "role": "model" if role == "assistant" else "user",
            "parts": [{"text": content}],
        }

    def _build_input(
        self,
        user: str,
        system: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the full Gemini model input."""
        # the system prompt is passed separately as part of the request config
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
        """Generate a model response from the Google Gemini API."""
        # built-in tools (google search, code execution, url context, etc.) are
        # deliberately left unset, so they remain off by default
        config = types.GenerateContentConfig(
            system_instruction=system or self._system,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            thinking_config=(
                types.ThinkingConfig(include_thoughts=True)
                if self._enable_reasoning
                else None
            ),
        )
        response = self._client.models.generate_content(
            model=model,
            contents=cast(types.ContentListUnionDict, input),
            config=config,
        )
        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate else None
        parts = (content.parts if content else None) or []

        # chain-of-thought from thinking-enabled models; None otherwise
        reasoning_parts = [
            part.text for part in parts if part.thought and part.text is not None
        ]
        reasoning = "\n".join(reasoning_parts) if reasoning_parts else None

        text_parts = [
            part.text for part in parts if not part.thought and part.text is not None
        ]
        return "\n".join(text_parts), reasoning
