"""Test the llm module."""

import pytest

from llm_cgr import (
    Anthropic_LLM,
    OpenAI_LLM,
    TogetherAI_LLM,
    generate,
    generate_bool,
    generate_list,
    get_llm,
)


def test_generate(model):
    """
    Test the generate method.
    """
    user = "What is the capital of France?"
    response = generate(user=user, model=model)
    assert isinstance(response, str)
    assert len(response) > 0


def test_generate_list(model):
    """
    Test the generate_list method.
    """
    user = "List the first 5 prime numbers."
    response = generate_list(user=user, model=model)
    assert isinstance(response, list)
    assert len(response) > 0
    assert all(isinstance(item, str) for item in response)


def test_generate_bool(model):
    """
    Test the generate_bool method.
    """
    user = "Is the sky blue?"
    response = generate_bool(user=user, model=model)
    assert isinstance(response, bool)
    assert response in {True, False}
    assert response is not None


def test_chat_flow(model):
    """
    Test the chat flow method.
    """
    llm = get_llm(model=model)

    response = llm.chat(user="Hello, can you help me with code?")
    assert isinstance(response, str)
    assert len(response) > 0

    response = llm.chat(user="Great, how can you help me?")
    assert isinstance(response, str)
    assert len(response) > 0

    history = llm.history
    assert isinstance(history, list)
    assert len(history) == 4
    assert all(isinstance(item, dict) for item in history)
    assert all(len(item) == 2 for item in history)

    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello, can you help me with code?"

    assert history[1]["role"] == "assistant"
    assert len(history[1]["content"]) > 0

    assert history[2]["role"] == "user"
    assert history[2]["content"] != "Hello, can you help me with code?"

    assert history[3]["role"] == "assistant"
    assert len(history[3]["content"]) > 0


def test_no_model():
    """
    Test the get_llm function without a model.
    """
    llm = OpenAI_LLM()
    with pytest.raises(ValueError, match="Model must be specified for OpenAI API."):
        llm.generate(user="What is the capital of Denmark?")

    llm = TogetherAI_LLM()
    with pytest.raises(ValueError, match="Model must be specified for TogetherAI API."):
        llm.generate(user="What is the capital of Canada?")

    llm = Anthropic_LLM()
    with pytest.raises(
        ValueError, match="Model must be specified for Anthropic Claude API."
    ):
        llm.generate(user="What is the capital of Portugal?")


def test_build_input():
    """
    Test the _build_input method.
    """
    llm = OpenAI_LLM()
    input_data = llm._build_input(user="What is the capital of Denmark?")
    assert isinstance(input_data, list)
    assert len(input_data) == 1
    assert input_data[0] == {
        "role": "user",
        "content": "What is the capital of Denmark?",
    }

    llm = TogetherAI_LLM()
    input_data = llm._build_input(user="What is the capital of Canada?")
    assert isinstance(input_data, list)
    assert len(input_data) == 1
    assert input_data[0] == {
        "role": "user",
        "content": "What is the capital of Canada?",
    }

    llm = Anthropic_LLM()
    input_data = llm._build_input(user="What is the capital of Portugal?")
    assert isinstance(input_data, list)
    assert len(input_data) == 1
    assert input_data[0] == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is the capital of Portugal?",
            }
        ],
    }
