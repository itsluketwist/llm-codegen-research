"""API utilities for interfacing with the completion models."""

from llm_cgr.defaults import DEFAULT_MODEL
from llm_cgr.query.completion import (
    AnthropicCompletionAPI,
    OpenAICompletionAPI,
    TogetherCompletionAPI,
)
from llm_cgr.query.prompts import BASE_SYSTEM_PROMPT, LIST_SYSTEM_PROMPT
from llm_cgr.query.protocol import CompletionProtocol


def get_client(model: str) -> CompletionProtocol:
    """
    Initialise the correct completion interface for the given model.
    """
    if "claude" in model:
        return AnthropicCompletionAPI(model=model)

    if "gpt" in model or "o1" in model:
        return OpenAICompletionAPI(model=model)

    return TogetherCompletionAPI(model=model)


def quick_complete(
    user: str,
    system: str = BASE_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Simple function to quickly prompt a model for a response.
    """
    client = get_client(model=model)

    result = client.complete(
        user=user,
        system=system,
    )
    return result[0]


def query_list(
    user: str,
    system: str = LIST_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """
    Simple function to quickly prompt a model for a list of words.
    """
    _response = quick_complete(
        user=user,
        system=system,
        model=model,
    )

    # sometimes the LLM will return a code block with the list inside it
    if _response.startswith("```python"):
        _response = _response.split("```python")[1]
    if _response.endswith("```"):
        _response = _response.split("```")[0]
    _response = _response.strip()

    try:
        _list = eval(_response)
    except Exception as e:
        print(f"Error evaluating response.\nresponse: {_response}\nexception: {e}")
        _list = []

    if not isinstance(_list, list):
        print(f"Error querying list. Response is not a list: {_list}")
        _list = []

    if any(not isinstance(item, str) for item in _list):
        print(f"Error querying list. Response contains non-string items: {_list}")
        _list = []

    return _list
