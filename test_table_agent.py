import os
import pytest
import shutil
import pandas as pd

from ai_acr import graph  # replace with your actual file name

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
TEST_IMAGE = "table.png"  # Make sure this exists


@pytest.fixture(autouse=True)
def clean_up_generated_files():
    # Runs before and after each test
    yield
    for file in os.listdir("."):
        if file.startswith("table_") or file.startswith("ocr_table"):
            os.remove(file)


def test_extract_html_tables():
    result = graph.invoke({
        "input": WIKI_URL,
        "format": "csv"
    })

    assert "table" in result["output"].lower()
    assert "files" in result
    for path in result["files"]:
        assert os.path.exists(path)
        df = pd.read_csv(path)
        assert not df.empty


def test_extract_ocr_table_from_image():
    if not os.path.exists(TEST_IMAGE):
        pytest.skip("OCR test image not found.")

    result = graph.invoke({
        "input": TEST_IMAGE,
        "format": "csv"
    })

    assert "ocr" in result["output"].lower()
    assert "files" in result
    assert result["files"][0].endswith(".csv")
    assert os.path.exists(result["files"][0])


def test_invalid_input_path():
    result = graph.invoke({
        "input": "not_a_valid_path_or_url",
        "format": "csv"
    })

    assert "unsupported" in result["output"].lower()


def test_unsupported_format_fallback():
    result = graph.invoke({
        "input": WIKI_URL,
        "format": "pdf"  # not supported
    })

    # Should fallback to default CSV
    assert "table" in result["output"].lower()
    assert any(f.endswith(".csv") for f in result.get("files", []))
