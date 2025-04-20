"""Protocol for the completion API of an LLM service."""

from typing import Protocol


class CompletionProtocol(Protocol):
    """
    Protocol that describes how to access the completion API of an LLM service.
    """

    def complete(
        self,
        user: str,
        system: str,
        model: str,
        n: int = 1,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> list[str]:
        pass
