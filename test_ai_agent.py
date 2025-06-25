import os
import pytest
from testing import (
    handle_summary,
    handle_text,
    handle_image,
    handle_webpage
)

# Use a short placeholder paragraph for summary/text gen
SHORT_TEXT = """
Python is a high-level programming language widely used for web development, automation, and AI applications.
"""

WIKI_URL = "https://en.wikipedia.org/wiki/Python_(programming_language)"


def test_handle_summary():
    state = {"input": f"Summarize this: {SHORT_TEXT}"}
    result = handle_summary(state)

    assert "input" in result
    assert "output" in result
    assert isinstance(result["output"], str)
    assert 20 < len(result["output"]) < 500


def test_handle_text_generation():
    state = {"input": "Explain how neural networks work"}
    result = handle_text(state)

    assert "input" in result
    assert "output" in result
    assert isinstance(result["output"], str)
    assert len(result["output"]) > 20


def test_handle_image_generation():
    state = {"input": "A robot playing chess in a cyberpunk city"}
    result = handle_image(state)

    assert "input" in result
    assert "output" in result
    assert os.path.exists("generated.png")
    assert result["output"].endswith("generated.png")

    # Cleanup image
    os.remove("generated.png")


def test_handle_webpage_summary():
    state = {"input": WIKI_URL}
    result = handle_webpage(state)

    assert "input" in result
    assert "output" in result
    assert isinstance(result["output"], str)
    assert len(result["output"]) > 20


def test_handle_webpage_invalid_url():
    state = {"input": "https://thisurldoesnotexist.openai404"}
    result = handle_webpage(state)

    assert "input" in result
    assert "output" in result
    assert "error" in result["output"].lower()
