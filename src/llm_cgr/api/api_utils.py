"""API utilities for interfacing with the completion models."""

from llm_cgr.api.completion import (
    AnthropicCompletionAPI,
    OpenAICompletionAPI,
    TogetherCompletionAPI,
)
from llm_cgr.api.prompts import BASE_SYSTEM_PROMPT
from llm_cgr.api.protocol import CompletionProtocol
from llm_cgr.md import Markdown


def get_client(model: str) -> CompletionProtocol:
    """
    Initialise the correct completion interface for the given model.
    """
    if "claude" in model:
        return AnthropicCompletionAPI()

    if "gpt" in model or "o1" in model:
        return OpenAICompletionAPI()

    return TogetherCompletionAPI()


def quick_complete(
    user: str,
    system: str = BASE_SYSTEM_PROMPT,
    model: str = "gpt-4o-mini",
) -> Markdown:
    """
    Simple function to quickly prompt a model for a response.
    """
    client = get_client(model=model)

    result = client.complete(
        user=user,
        system=system,
        model=model,
    )
    return Markdown(text=result[0])
