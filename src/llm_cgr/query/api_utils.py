"""API utilities for interfacing with the generation models."""

from typing import Literal

from llm_cgr.defaults import DEFAULT_MODEL
from llm_cgr.query.generate import (
    AnthropicGenerationAPI,
    OpenAIGenerationAPI,
    TogetherGenerationAPI,
)
from llm_cgr.query.prompts import (
    BASE_SYSTEM_PROMPT,
    CODE_SYSTEM_PROMPT,
    LIST_SYSTEM_PROMPT,
)
from llm_cgr.query.protocol import GenerationProtocol


def get_client(
    model: str,
    type: Literal["base", "code", "list"] | None = None,
) -> GenerationProtocol:
    """
    Initialise the correct generation interface for the given model.
    """
    _system = None
    match type:
        case "base":
            _system = BASE_SYSTEM_PROMPT
        case "code":
            _system = CODE_SYSTEM_PROMPT
        case "list":
            _system = LIST_SYSTEM_PROMPT

    if "claude" in model:
        return AnthropicGenerationAPI(model=model, system=_system)

    if "gpt" in model or "o1" in model:
        return OpenAIGenerationAPI(model=model, system=_system)

    return TogetherGenerationAPI(model=model, system=_system)


def quick_generate(
    user: str,
    type: Literal["base", "code", "list"] | None = None,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float | None = None,
) -> str:
    """
    Simple function to quickly prompt a model for a response.
    """
    client = get_client(model=model, type=type)

    [result] = client.generate(
        user=user,
        system=system,
        temperature=temperature,
    )
    return result


def query_list(
    user: str,
    system: str = LIST_SYSTEM_PROMPT,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """
    Simple function to quickly prompt a model for a list of words.
    """
    _response = quick_generate(
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
