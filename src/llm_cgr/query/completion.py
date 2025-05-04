"""Classes for access to completion APIs of LLM services."""

import anthropic
import openai
import together

from llm_cgr.query.prompts import BASE_SYSTEM_PROMPT


class OpenAICompletionAPI:
    """
    Class to access to OpenAI's API.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model
        self._client = openai.OpenAI()

    def complete(
        self,
        user: str,
        system: str = BASE_SYSTEM_PROMPT,
        model: str | None = None,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[str]:
        """
        Method to get the completion result of a prompt from an LLM model.

        Returns
        -------
        The textual response to the prompt.
        """
        response = self._client.chat.completions.create(
            model=model or self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            n=n,
            temperature=temperature or openai.NOT_GIVEN,
            max_tokens=max_tokens or openai.NOT_GIVEN,
        )
        choices = [c.message.content for c in response.choices]
        return choices


class TogetherCompletionAPI:
    """
    Class to access TogetherAI's API.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model
        self._client = together.Together()

    def complete(
        self,
        user: str,
        system: str = BASE_SYSTEM_PROMPT,
        model: str | None = None,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[str]:
        response = self._client.chat.completions.create(
            model=model or self._model,
            system=system,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            n=n,
            temperature=temperature,
            max_tokens=max_tokens or 1000,
        )
        choices = [c.message.content for c in response.choices]
        return choices


class AnthropicCompletionAPI:
    """
    Class to access Anthropic's Claude API.
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model
        self._client = anthropic.Anthropic()

    def complete(
        self,
        user: str,
        system: str = BASE_SYSTEM_PROMPT,
        model: str | None = None,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[str]:
        response = self._client.messages.create(
            model=model or self._model,
            temperature=temperature or anthropic.NOT_GIVEN,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user,
                        },
                    ],
                },
            ],
            max_tokens=max_tokens or 1000,
        )
        return [response.content[0].text]
