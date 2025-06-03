"""Module for LLM client initialisation."""

from llm_cgr.llm.clients.anthropic import Anthropic_LLM
from llm_cgr.llm.clients.base import Base_LLM
from llm_cgr.llm.clients.mistral import Mistral_LLM
from llm_cgr.llm.clients.openai import OpenAI_LLM
from llm_cgr.llm.clients.protocol import GenerationProtocol
from llm_cgr.llm.clients.together import TogetherAI_LLM


def get_llm(
    model: str,
    system: str | None = None,
) -> GenerationProtocol:
    """
    Initialise the correct LLM client for the given model.
    """
    if "claude" in model:
        return Anthropic_LLM(model=model, system=system)

    if "gpt" in model or "o1" in model:
        return OpenAI_LLM(model=model, system=system)

    if "mistral" in model:
        return Mistral_LLM(model=model, system=system)

    return TogetherAI_LLM(model=model, system=system)


__all__ = [
    "Anthropic_LLM",
    "Base_LLM",
    "GenerationProtocol",
    "OpenAI_LLM",
    "TogetherAI_LLM",
    "Mistral_LLM",
    "get_llm",
]
