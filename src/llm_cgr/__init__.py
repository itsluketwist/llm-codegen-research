"""Package initialisation, exporting all public APIs."""

from llm_cgr import analyse
from llm_cgr.analyse import CodeBlock, CodeData, Markdown, analyse_code
from llm_cgr.decorators import experiment
from llm_cgr.enums import OptionsEnum
from llm_cgr.json_utils import load_json, load_jsonl, save_json, save_jsonl
from llm_cgr.timeout import TimeoutException, timeout


# core exports, always available
__all__ = [
    # modules
    "analyse",
    # analyse members
    "CodeBlock",
    "CodeData",
    "Markdown",
    "analyse_code",
    # decorators
    "experiment",
    # enums
    "OptionsEnum",
    # json utilities
    "load_json",
    "load_jsonl",
    "save_json",
    "save_jsonl",
    # timeout
    "TimeoutException",
    "timeout",
]

# llm members are only available if the 'api' extra is installed
try:
    from llm_cgr import llm
    from llm_cgr.llm import (
        BASE_SYSTEM_PROMPT,
        BOOL_SYSTEM_PROMPT,
        CODE_SYSTEM_PROMPT,
        LIST_SYSTEM_PROMPT,
        Anthropic_LLM,
        Base_LLM,
        DeepSeek_LLM,
        GenerationProtocol,
        Mistral_LLM,
        OpenAI_LLM,
        TogetherAI_LLM,
        generate,
        generate_bool,
        generate_list,
        get_llm,
    )

    __all__ += [
        "llm",
        "BASE_SYSTEM_PROMPT",
        "BOOL_SYSTEM_PROMPT",
        "CODE_SYSTEM_PROMPT",
        "LIST_SYSTEM_PROMPT",
        "Anthropic_LLM",
        "Base_LLM",
        "DeepSeek_LLM",
        "GenerationProtocol",
        "Mistral_LLM",
        "OpenAI_LLM",
        "TogetherAI_LLM",
        "generate",
        "generate_bool",
        "generate_list",
        "get_llm",
    ]

except ImportError:
    pass
