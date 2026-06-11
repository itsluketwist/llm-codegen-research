"""Tests for Mistral reasoning model support."""

import pytest

from llm_cgr.llm.clients.mistral import Mistral_LLM


# mark all tests in this file as api tests, so they can be excluded in ci
pytestmark = pytest.mark.api

# standard model returns no chain-of-thought; magistral models do
CHAT_MODEL = "mistral-small-latest"
REASONER_MODEL = "magistral-medium-latest"

USER_PROMPT = "How many r's are in 'strawberry'?"


def test_generate_no_reasoning():
    """
    Test that generate returns plain strings when enable_reasoning is False (default).
    """
    llm = Mistral_LLM(model=CHAT_MODEL)
    results = llm.generate(user=USER_PROMPT)

    assert isinstance(results, list)
    assert len(results) == 1
    # result should be a plain string, not a tuple
    assert isinstance(results[0], str)
    assert len(results[0]) > 0


def test_generate_with_reasoning_returns_tuples():
    """
    Test that generate returns (response, reasoning) tuples when enable_reasoning is True.
    """
    llm = Mistral_LLM(model=REASONER_MODEL, enable_reasoning=True)
    results = llm.generate(user=USER_PROMPT)

    assert isinstance(results, list)
    assert len(results) == 1

    response, reasoning = results[0]

    # response should be a non-empty string
    assert isinstance(response, str)
    assert len(response) > 0

    # the reasoning model should produce chain-of-thought
    assert isinstance(reasoning, str)
    assert len(reasoning) > 0


def test_generate_non_reasoning_model_has_no_reasoning():
    """
    Test that a standard (non-reasoning) model returns None for reasoning even when enabled.
    """
    llm = Mistral_LLM(model=CHAT_MODEL, enable_reasoning=True)
    results = llm.generate(user=USER_PROMPT)

    response, reasoning = results[0]

    assert isinstance(response, str)
    assert len(response) > 0
    # mistral-small-latest does not produce reasoning content
    assert reasoning is None


def test_chat_no_reasoning():
    """
    Test that chat returns a plain string and history has no reasoning_content
    when enable_reasoning is False (default).
    """
    llm = Mistral_LLM(model=CHAT_MODEL)
    response = llm.chat(user=USER_PROMPT)

    assert isinstance(response, str)
    assert len(response) > 0

    # history entries should each have exactly role and content
    history = llm.history
    assert all("reasoning_content" not in msg for msg in history)


def test_chat_with_reasoning_returns_tuple():
    """
    Test that chat returns a (response, reasoning) tuple when enable_reasoning is True.
    """
    llm = Mistral_LLM(model=REASONER_MODEL, enable_reasoning=True)
    result = llm.chat(user=USER_PROMPT)

    assert isinstance(result, tuple)
    response, reasoning = result

    assert isinstance(response, str)
    assert len(response) > 0

    assert isinstance(reasoning, str)
    assert len(reasoning) > 0


def test_chat_reasoning_stored_in_history():
    """
    Test that reasoning is stored on the assistant history entry when enable_reasoning is True.
    """
    llm = Mistral_LLM(model=REASONER_MODEL, enable_reasoning=True)
    llm.chat(user=USER_PROMPT)

    history = llm.history
    # find the assistant message
    assistant_msgs = [msg for msg in history if msg["role"] == "assistant"]
    assert len(assistant_msgs) == 1

    assistant_msg = assistant_msgs[0]
    assert "reasoning_content" in assistant_msg
    assert isinstance(assistant_msg["reasoning_content"], str)
    assert len(assistant_msg["reasoning_content"]) > 0


def test_chat_multi_turn_reasoning_stored_per_turn():
    """
    Test that reasoning is captured and stored for each turn in a multi-turn chat.
    """
    llm = Mistral_LLM(model=REASONER_MODEL, enable_reasoning=True)

    llm.chat(user="What is 2 + 2?")
    llm.chat(user="And what is that result multiplied by 3?")

    history = llm.history
    assistant_msgs = [msg for msg in history if msg["role"] == "assistant"]
    assert len(assistant_msgs) == 2

    # both assistant turns should have reasoning attached
    for msg in assistant_msgs:
        assert "reasoning_content" in msg
        assert isinstance(msg["reasoning_content"], str)
        assert len(msg["reasoning_content"]) > 0
