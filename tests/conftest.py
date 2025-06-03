"""Fixtures used for testing the project."""

import shutil
import tempfile

import pytest


@pytest.fixture(
    params=[
        "gpt-4.1-nano-2025-04-14",
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "claude-3-5-haiku-20241022",
        "mistral-small-2503",
    ],
)
def model(request):
    """
    Fixture to provide different models for testing.
    """
    return request.param


@pytest.fixture
def openai_model():
    """
    Fixture to provide a specific GPT model for testing.
    """
    return "gpt-4o-mini-2024-07-18"


@pytest.fixture
def temp_dir():
    """
    Fixture to create a temporary directory for testing.

    Yields the path to the temporary directory.
    """
    path = tempfile.mkdtemp()  # create
    try:
        yield path

    finally:
        shutil.rmtree(path)  # cleanup
