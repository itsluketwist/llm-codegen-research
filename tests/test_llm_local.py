"""Test the LLM classes locally without connecting to the APIs."""

from unittest.mock import patch

import pytest

from llm_cgr import (
    Anthropic_LLM,
    Mistral_LLM,
    OpenAI_LLM,
    TogetherAI_LLM,
    generate_bool,
    generate_list,
)


def test_no_model():
    """
    Test the get_llm function without a model.
    """
    llm = OpenAI_LLM()
    with pytest.raises(ValueError, match="Model must be specified for LLM APIs."):
        llm.generate(user="What is the capital of Canada?")

    llm = TogetherAI_LLM()
    with pytest.raises(ValueError, match="Model must be specified for LLM APIs."):
        llm.generate(user="What is the capital of Brazil?")

    llm = Anthropic_LLM()
    with pytest.raises(ValueError, match="Model must be specified for LLM APIs."):
        llm.chat(user="What is the capital of Ireland?")

    llm = Mistral_LLM()
    with pytest.raises(ValueError, match="Model must be specified for LLM APIs."):
        llm.chat(user="What is the capital of Hawaii?")


def test_build_input():
    """
    Test the _build_input method.
    """
    llm = OpenAI_LLM()
    input_data = llm._build_input(user="What is the capital of Canada?")
    assert isinstance(input_data, list)
    assert len(input_data) == 1
    assert input_data[0] == {
        "role": "user",
        "content": "What is the capital of Canada?",
    }

    llm = TogetherAI_LLM()
    input_data = llm._build_input(user="What is the capital of Brazil?")
    assert isinstance(input_data, list)
    assert len(input_data) == 1
    assert input_data[0] == {
        "role": "user",
        "content": "What is the capital of Brazil?",
    }

    llm = Anthropic_LLM()
    input_data = llm._build_input(user="What is the capital of Ireland?")
    assert isinstance(input_data, list)
    assert len(input_data) == 1
    assert input_data[0] == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is the capital of Ireland?",
            }
        ],
    }

    llm = Mistral_LLM()
    input_data = llm._build_input(user="What is the capital of Hawaii?")
    assert isinstance(input_data, list)
    assert len(input_data) == 1
    assert input_data[0] == {
        "role": "user",
        "content": "What is the capital of Hawaii?",
    }


@pytest.mark.parametrize(
    "response,error",
    [
        (
            "This is not a python list.",
            "Error evaluating response. Response: This is not a python list.\n",
        ),
        (
            "{1, 2, 3}",
            "Error querying list. Response is not a list: {1, 2, 3}\n",
        ),
        (
            "['one', 2, 'three']",
            "Error querying list. Response contains non-string items: ['one', 2, 'three']\n",
        ),
    ],
)
def test_generate_list_errors(capfd, response, error):
    """
    Test the generate_list method with various error cases by mocking out the generate function.
    """

    with patch("llm_cgr.llm.generate.generate") as mock_generate:
        # mock the response
        mock_generate.return_value = response
        response = generate_list(user="Give me an error I guess?")

        # check no return
        assert response == []

        # just printed error
        captured = capfd.readouterr()
        assert captured.out == error


@pytest.mark.parametrize(
    "response,error",
    [
        (
            "This is not a python boolean.",
            "Error evaluating response. Response: This is not a python boolean.\n",
        ),
        (
            "42",
            "Error querying boolean. Response is not a boolean: 42\n",
        ),
    ],
)
def test_generate_bool_errors(capfd, response, error):
    """
    Test the generate_bool method with an error case by mocking out the generate function.
    """
    with patch("llm_cgr.llm.generate.generate") as mock_generate:
        # mock the response
        mock_generate.return_value = response
        response = generate_bool(user="Give me an error I guess?")

        # check no return
        assert response is False

        # just printed error
        captured = capfd.readouterr()
        assert captured.out.startswith(error)
