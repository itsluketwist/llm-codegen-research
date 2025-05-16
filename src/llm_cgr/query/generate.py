"""Classes for access to generation APIs of LLM services."""

import anthropic
import openai
import together

from llm_cgr.defaults import DEFAULT_MAX_TOKENS


class OpenAIGenerationAPI:
    """
    Class to access the OpenAI API.
    """

    def __init__(
        self,
        model: str | None = None,
        system: str | None = None,
    ) -> None:
        self._model = model
        self._system = system or openai.NOT_GIVEN
        self._client = openai.OpenAI()

    def generate(
        self,
        user: str,
        system: str | None = None,
        model: str | None = None,
        samples: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[str]:
        """
        Generate a model response from the OpenAI API.

        Returns
        -------
        The text response to the prompt.
        """
        _generations = []
        for _ in range(samples):
            response = self._client.responses.create(
                input=user,
                model=model or self._model,
                instructions=system or self._system,
                temperature=temperature or openai.NOT_GIVEN,
                max_output_tokens=max_tokens or openai.NOT_GIVEN,
            )
            _generations.append(response.output_text)

        return _generations


class TogetherGenerationAPI:
    """
    Class to access the TogetherAI API.
    """

    def __init__(
        self,
        model: str | None = None,
        system: str | None = None,
    ) -> None:
        self._model = model
        self._system = system
        self._client = together.Together()

    def generate(
        self,
        user: str,
        system: str | None = None,
        model: str | None = None,
        samples: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[str]:
        """
        Generate a model response from the TogetherAI API.

        Returns
        -------
        The text response to the prompt.
        """
        _input = [{"role": "user", "content": user}]
        _system = system or self._system
        if _system:
            _input = [{"role": "system", "content": _system}] + _input

        _generations = []
        for _ in range(samples):
            response = self._client.chat.completions.create(
                model=model or self._model,
                messages=_input,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            _generations.append(response.choices[0].message.content)

        return _generations


class AnthropicGenerationAPI:
    """
    Class to access the Anthropic Claude API.
    """

    def __init__(self, model: str | None = None, system: str | None = None) -> None:
        self._model = model
        self._system = system or anthropic.NOT_GIVEN
        self._client = anthropic.Anthropic()

    def generate(
        self,
        user: str,
        system: str | None = None,
        model: str | None = None,
        samples: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[str]:
        """
        Generate a model response from the Anthropic Claude API.

        Returns
        -------
        The text response to the prompt.
        """
        _generations = []
        for _ in range(samples):
            response = self._client.messages.create(
                model=model or self._model,
                system=system or self._system,
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
                temperature=temperature or anthropic.NOT_GIVEN,
                max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
            )
            _generations.append(response.content[0].text)

        return _generations
