from llm_cgr.api.api_utils import get_client, quick_complete
from llm_cgr.api.completion import (
    AnthropicCompletionAPI,
    OpenAICompletionAPI,
    TogetherCompletionAPI,
)
from llm_cgr.api.protocol import CompletionProtocol


__all__ = [
    "get_client",
    "quick_complete",
    "AnthropicCompletionAPI",
    "OpenAICompletionAPI",
    "TogetherCompletionAPI",
    "CompletionProtocol",
]
