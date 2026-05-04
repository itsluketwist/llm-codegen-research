"""OpenAI client subclass with an agentic tool-call loop."""

import json
from dataclasses import dataclass
from typing import Any, Callable, cast

import openai
from openai.types.responses import ResponseFunctionToolCall, ResponseInputItemParam

from llm_cgr.llm.clients.openai import OpenAI_LLM


# maximum tool-call rounds allowed within a single generate() or chat() call
MAX_TOOL_ITERATIONS: int = 5

# maximum total tool calls allowed across the lifetime of a client instance
MAX_TOOL_CALLS: int = 10


@dataclass
class Tool:
    """
    A tool (function) that the model can call during generation.

    Attributes:
        name: The function name the model uses to call this tool.
        description: Describes what the tool does; the model uses this
            to decide when to call it.
        parameters: A JSON schema dict describing the function's parameters.
        fn: The Python callable to invoke; must accept kwargs matching the
            schema and return a str result.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., str]


class OpenAI_Tool_LLM(OpenAI_LLM):
    """OpenAI client with an agentic tool-call loop.

    Tools are supplied at construction time and used for all subsequent
    generate() and chat() calls. The client handles the full loop internally:
    call the API, execute any tool calls, feed results back, repeat until the
    model produces a final text response.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_tool_iterations: int = MAX_TOOL_ITERATIONS,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ) -> None:
        """
        Initialise the OpenAI tool client.

        Requires the OPENAI_API_KEY environment variable to be set.
        max_tool_iterations caps tool-call rounds within a single request.
        max_tool_calls caps the cumulative total across all requests on this
        instance. When either limit is reached, the model is sent a message
        asking it to answer immediately without any further tool calls.
        """
        super().__init__(
            model=model,
            system=system,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        self._tools = tools
        self._max_tool_iterations = max_tool_iterations
        self._max_tool_calls = max_tool_calls
        # cumulative count of individual tool calls made by this instance
        self._tool_calls: int = 0

    @property
    def tool_calls(self) -> int:
        """Total number of tool calls made by this client since instantiation.

        Returns the cumulative count across all generate() and chat() calls.
        Tip: record the value before a call and subtract to get the count for
        that specific call.
        """
        return self._tool_calls

    def _build_tool_param(
        self,
        tool: Tool,
    ) -> dict[str, Any]:
        """Convert a Tool dataclass to the dict format the OpenAI Responses API expects."""
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }

    def _force_final_answer(
        self,
        current_input: list[Any],
        model: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> str:
        """Force the model to produce a text answer after a limit is reached.

        Appends a user message telling the model it has used all its allowed
        tool calls, then calls the API one final time without any tools so the
        model cannot make further calls.

        Returns the model's final text response.
        """
        # tell the model it must answer now — no more tool calls are allowed
        current_input.append(
            self._build_message(
                role="user",
                content=(
                    "You have reached the maximum number of tool calls allowed. "
                    "Please provide your final answer now based on the information "
                    "you have gathered, without calling any more tools."
                ),
            )
        )
        response = self._client.responses.create(
            input=cast(list[ResponseInputItemParam], current_input),
            model=model,
            temperature=temperature if temperature is not None else openai.omit,
            top_p=top_p if top_p is not None else openai.omit,
            max_output_tokens=max_tokens if max_tokens is not None else openai.omit,
            # no tools provided: the model cannot make further tool calls
        )
        return response.output_text

    def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> str:
        """Run the agentic tool-call loop for a single turn.

        Calls the OpenAI API in a loop, executing any tool calls the model
        requests, until the model produces a final text response or a limit is
        reached. Two limits apply:
          - max_tool_iterations: rounds allowed within this single call.
          - max_tool_calls: cumulative total across all calls on this instance.
        When either limit is hit, _force_final_answer() is called, which tells
        the model to answer immediately without making any further tool calls.

        Returns the final text response.
        """
        # convert Tool dataclasses to the API's function-tool format
        api_tools = [self._build_tool_param(t) for t in self._tools]

        # build a name -> Tool lookup map for fast dispatch during the loop
        tool_map = {t.name: t for t in self._tools}

        # shallow copy so intermediate tool-call scaffolding never mutates the
        # caller's message list (prevents corruption of the chat history).
        # typed as list[Any] so we can freely append both plain message dicts
        # and the richer tool-call dicts without fighting the type checker.
        current_input: list[Any] = list(messages)

        for _ in range(self._max_tool_iterations):
            response = self._client.responses.create(
                input=cast(list[ResponseInputItemParam], current_input),
                model=model,
                temperature=temperature if temperature is not None else openai.omit,
                top_p=top_p if top_p is not None else openai.omit,
                max_output_tokens=max_tokens if max_tokens is not None else openai.omit,
                tools=cast(Any, api_tools),
            )

            # collect any function calls the model requested in this response
            function_calls = [
                item for item in response.output if item.type == "function_call"
            ]

            # no tool calls means the model has produced its final text answer
            if not function_calls:
                return response.output_text

            # check the overall cumulative limit before processing these calls.
            # if adding them would exceed the limit, force a final answer now
            # without executing any of the pending tool calls.
            if self._tool_calls + len(function_calls) > self._max_tool_calls:
                return self._force_final_answer(
                    current_input=current_input,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )

            # increment the cumulative counter; parallel calls count individually
            self._tool_calls += len(function_calls)

            # process each tool call: the OpenAI Responses API requires that the
            # function_call item appears in the next input before its matching
            # function_call_output item
            for _call in function_calls:
                # cast to the concrete type so we can access .call_id/.name/.arguments
                call = cast(ResponseFunctionToolCall, _call)

                # append the function_call itself so the model sees what it called
                current_input.append(
                    {
                        "type": "function_call",
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                )

                # deserialise the model's json argument string and call the local fn
                kwargs = json.loads(call.arguments)
                result = tool_map[call.name].fn(**kwargs)

                # append the result so the model can read it on the next turn
                current_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": result,
                    }
                )

            # loop continues: enriched input is sent back to the model

        # max_tool_iterations exhausted — force the model to answer now
        return self._force_final_answer(
            current_input=current_input,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def generate(
        self,
        user: str,
        system: str | None = None,
        model: str | None = None,
        samples: int = 1,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> list[str]:
        """Generate model responses via the agentic tool-call loop."""
        _model = model or self._model
        if _model is None:
            raise ValueError("Model must be specified for LLM APIs.")

        messages = self._build_input(
            user=user,
            system=system or self._system,
        )

        _generations = []
        for _ in range(samples):
            result = self._run_tool_loop(
                messages=messages,
                model=_model,
                temperature=temperature or self._temperature,
                top_p=top_p or self._top_p,
                max_tokens=max_tokens or self._max_tokens,
            )
            _generations.append(result)

        return _generations

    def chat(
        self,
        user: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run a chat turn via the agentic tool-call loop.

        Manages self._history identically to the base class — only the final
        text response is appended, not intermediate tool-call scaffolding.
        """
        _model = model or self._model
        if _model is None:
            raise ValueError("Model must be specified for LLM APIs.")

        if self._history is None:
            self._history = self._build_input(
                user=user,
                system=system or self._system,
            )
        else:
            self._history.append(
                self._build_message(
                    role="user",
                    content=user,
                )
            )

        # _run_tool_loop operates on a shallow copy of self._history, so
        # intermediate tool-call items never appear in the chat history
        response = self._run_tool_loop(
            messages=self._history,
            model=_model,
            temperature=temperature or self._temperature,
            top_p=top_p or self._top_p,
            max_tokens=max_tokens or self._max_tokens,
        )

        self._history.append(
            self._build_message(
                role="assistant",
                content=response,
            )
        )
        return response
