"""Test the OpenAI_Tool_LLM agentic tool-call loop."""

import pytest

from llm_cgr import OpenAI_Tool_LLM, Tool


# mark all tests in this file as api tests, so they can be excluded in ci
pytestmark = pytest.mark.api


def test_tool_call_generate(openai_model):
    """
    Test that the OpenAI tool client runs the agentic loop and returns the
    correct answer via the tool.

    Uses an addition tool: the model must call it to get the answer, so we can
    verify a real tool call happened (the model cannot guess what our local
    function returns without calling it).
    """

    def add(a: int, b: int) -> str:
        """Add two integers and return the result as a string."""
        return str(a + b)

    add_tool = Tool(
        name="add",
        description="Add two integers together and return the result.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "The first integer."},
                "b": {"type": "integer", "description": "The second integer."},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        fn=add,
    )

    llm = OpenAI_Tool_LLM(tools=[add_tool], model=openai_model)
    responses = llm.generate(
        user="Use the add tool to compute 3 + 4. What is the result?"
    )

    assert isinstance(responses, list)
    assert len(responses) == 1

    result = responses[0]
    assert isinstance(result, str)
    assert len(result) > 0
    # the correct sum proves the tool was actually called
    assert "7" in result
    assert llm.tool_calls >= 1


def test_tool_call_chat(openai_model):
    """
    Test that tool calls work correctly in a chat session, and that
    intermediate tool-call scaffolding does not corrupt the chat history.
    """

    def multiply(a: int, b: int) -> str:
        """Multiply two integers and return the result as a string."""
        return str(a * b)

    multiply_tool = Tool(
        name="multiply",
        description="Multiply two integers together and return the result.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "The first integer."},
                "b": {"type": "integer", "description": "The second integer."},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        fn=multiply,
    )

    llm = OpenAI_Tool_LLM(tools=[multiply_tool], model=openai_model)
    response = llm.chat(
        user="Use the multiply tool to compute 6 * 7. What is the result?"
    )

    assert isinstance(response, str)
    assert "42" in response

    # history should only contain the user turn and the final assistant response;
    # no intermediate function_call or function_call_output items should have leaked in
    history = llm.history
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert llm.tool_calls >= 1
