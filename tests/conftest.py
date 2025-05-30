"""Fixtures used for testing the project."""

import shutil
import tempfile

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
