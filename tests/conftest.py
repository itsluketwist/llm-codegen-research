"""Fixtures used for testing the project."""

import pytest


@pytest.fixture(
    params=[
        "gpt-4.1-nano-2025-04-14",
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        pytest.param(
            "claude-3-5-haiku-20241022",
            marks=pytest.mark.skip(reason="expensive"),
        ),
    ],
)
def model(request):
    """
    Fixture to provide different models for testing.
    """
    return request.param
