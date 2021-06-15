import pytest
from innovation_sweet_spots.analysis.analysis_utils import (
    create_documents,
    create_documents_from_dataframe,
    preprocess_text,
)
import pandas as pd

### Preprocessing
def test_create_documents():
    assert list(create_documents((["one", "two"], ["cat", "dogs"]))) == [
        "one cat",
        "two dogs",
    ]
    with pytest.raises(ValueError):
        create_documents((["one"], ["cat", "dogs"]))


def test_create_documents_from_dataframe():
    mock_table = pd.DataFrame(
        data={
            "numbers": ["one", "two"],
            "fruit": ["APPLE", "oranges"],
            "animals": ["cat", None],
        }
    )
    docs = create_documents_from_dataframe(mock_table, columns=["numbers", "fruit"])
    assert docs == ["one apple", "two oranges"]
    # Test a case with null values
    docs = create_documents_from_dataframe(mock_table, columns=["numbers", "animals"])
    assert docs == ["one cat", "two"]


def test_preprocess_text():
    assert preprocess_text("USA") == "usa"
    assert preprocess_text("PEP8") == "pep8"
    assert preprocess_text("net-zero") == "net-zero"
