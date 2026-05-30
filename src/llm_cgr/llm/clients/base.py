"""Base class for LLM API clients."""

from abc import ABC, abstractmethod
from typing import Any


class Base_LLM(ABC):
    """Base class to access LLMs via their APIs."""

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
        Initialise the LLM client.

        When enable_reasoning is True, generate() and chat() include chain-of-thought
        alongside responses, and reasoning is stored in the chat history.
        """
        self._model = model
        self._system = system

        # default parameters
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens

        self._enable_reasoning = enable_reasoning
        self._history: list[dict[str, Any]] | None = None

    def generate(
        self,
        user: str,
        system: str | None = None,
        model: str | None = None,
        samples: int = 1,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> list[str] | list[tuple[str, str | None]]:
        """
        Generate model responses from the LLMs API.

        When enable_reasoning is True, returns a list of (response, reasoning) tuples.
        When False, returns a list of response strings.
        """
        _model = model or self._model
        if _model is None:
            raise ValueError("Model must be specified for LLM APIs.")

        messages = self._build_input(
            user=user,
            system=system or self._system,
        )

        _generations: list[Any] = []
        for _ in range(samples):
            response, reasoning = self._get_response(
                input=messages,
                model=_model,
                temperature=temperature or self._temperature,
                top_p=top_p or self._top_p,
                max_tokens=max_tokens or self._max_tokens,
            )
            if self._enable_reasoning:
                _generations.append((response, reasoning))
            else:
                _generations.append(response)

        return _generations

    def chat(
        self,
        user: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str | tuple[str, str | None]:
        """
        Generate a model response from the LLMs API, in the ongoing chat.

        When enable_reasoning is True, reasoning is stored in the history and the
        return value is a (response, reasoning) tuple instead of a plain string.
        """
        _model = model or self._model
        if _model is None:
            raise ValueError("Model must be specified for LLM APIs.")

        if self._history is None:
            # initialise the history
            self._history = self._build_input(
                user=user,
                system=system or self._system,
            )
        else:
            # or add the new message
            self._history.append(
                self._build_message(
                    role="user",
                    content=user,
                )
            )

        response, reasoning = self._get_response(
            input=self._history,
            system=system,
            model=_model,
            temperature=temperature or self._temperature,
            top_p=top_p or self._top_p,
            max_tokens=max_tokens or self._max_tokens,
        )

        # build the assistant history entry, attaching reasoning if present
        assistant_message = self._build_message(role="assistant", content=response)
        if self._enable_reasoning and reasoning is not None:
            assistant_message["reasoning_content"] = reasoning
        self._history.append(assistant_message)

        if self._enable_reasoning:
            return response, reasoning
        return response

    @property
    def history(self) -> list[dict[str, Any]]:
        """
        Get the chat history for this session.
        """
        return self._history or []

    @abstractmethod
    def _build_message(
        self,
        role: str,
        content: str,
    ) -> dict[str, Any]:
        """
        Build an LLM input message.
        """

    @abstractmethod
    def _build_input(
        self,
        user: str,
        system: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the full LLM input, with system and user messages if needed.
        """

    @abstractmethod
    def _get_response(
        self,
        model: str,
        input: list[dict[str, Any]],
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, str | None]:
        """
        Generate a model response from the LLM API.

        Returns a (response, reasoning) tuple; reasoning is None for models that
        do not produce chain-of-thought output.
        """
