"""Module for LLM client initialisation."""

from llm_cgr.llm.clients.anthropic import Anthropic_LLM
from llm_cgr.llm.clients.base import Base_LLM
from llm_cgr.llm.clients.deepseek import DeepSeek_LLM
from llm_cgr.llm.clients.mistral import Mistral_LLM
from llm_cgr.llm.clients.nscale import Nscale_LLM
from llm_cgr.llm.clients.openai import OpenAI_LLM
from llm_cgr.llm.clients.openai_tool import OpenAI_Tool_LLM, Tool
from llm_cgr.llm.clients.protocol import GenerationProtocol
from llm_cgr.llm.clients.together import TogetherAI_LLM


PROVIDER_MAP: dict[str, type[Base_LLM]] = {
    "anthropic": Anthropic_LLM,
    "deepseek": DeepSeek_LLM,
    "mistral": Mistral_LLM,
    "openai": OpenAI_LLM,
    "together": TogetherAI_LLM,
    "nscale": Nscale_LLM,
}


def get_llm(
    model: str,
    system: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    provider: str | None = None,
    tools: list[Tool] | None = None,
) -> GenerationProtocol:
    """
    Initialise the correct LLM client for the given model.

    If tools are provided, returns an OpenAI_Tool_LLM instance. Tool calls
    are currently only supported for OpenAI models.
    """
    llm_class: type[Base_LLM]
    if provider is not None:
        llm_class = PROVIDER_MAP[provider]
    elif "claude" in model:
        llm_class = Anthropic_LLM
    elif "gpt" in model or model.startswith("o"):
        llm_class = OpenAI_LLM
    elif "tral" in model:
        llm_class = Mistral_LLM
    elif "deepseek" in model:
        llm_class = DeepSeek_LLM
    else:
        llm_class = TogetherAI_LLM

    # if tools are requested, use the tool-enabled subclass (openai only for now)
    if tools is not None:
        if llm_class is not OpenAI_LLM:
            raise NotImplementedError(
                "Tool calls are only supported for OpenAI models."
            )
        return OpenAI_Tool_LLM(
            tools=tools,
            model=model,
            system=system,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    return llm_class(
        model=model,
        system=system,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


__all__ = [
    "Anthropic_LLM",
    "Base_LLM",
    "DeepSeek_LLM",
    "GenerationProtocol",
    "OpenAI_LLM",
    "OpenAI_Tool_LLM",
    "TogetherAI_LLM",
    "Mistral_LLM",
    "Tool",
    "get_llm",
]
