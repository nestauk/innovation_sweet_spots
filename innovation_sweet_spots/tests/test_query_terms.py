from innovation_sweet_spots.analysis.query_terms import *
from numpy.testing import assert_array_equal, assert_equal


def test_token_list_to_string():
    tokens = ["best", "heat_pump"]
    assert token_list_to_string(tokens) == " best heat pump "
    assert (
        token_list_to_string(tokens, within_token_separator="-") == " best heat_pump "
    )
    assert token_list_to_string(tokens, padding="") == "best heat pump"
    assert token_list_to_string(["best", "heat-pump"], "-") == " best heat pump "


def test_is_search_term_present():
    documents = ["heat pump", "pumps"]
    assert is_search_term_present("heat", documents) == [True, False]
    assert is_search_term_present("pump", documents) == [True, True]
    assert is_search_term_present("hydrogen", documents) == [False, False]


def test_get_matching_documents():
    documents = ["heat pump", "pumps"]
    get_matching_documents(documents, [True, False]) == ["heat pump"]
    get_matching_documents(documents, [False, True]) == ["pumps"]
    get_matching_documents(documents, [False, False]) == []


def test_find_documents_with_terms():
    documents = ["heat pump", "pumps"]
    assert_array_equal(find_documents_with_terms(["pump"], documents), [True, True])
    assert_array_equal(
        find_documents_with_terms(["heat", "pump"], documents), [True, False]
    )
    assert_array_equal(
        find_documents_with_terms(["heating"], documents), [False, False]
    )


def test_find_documents_with_set_of_terms():
    documents = ["heat pump", "hydrogen heating", "hydrogen"]
    assert_equal(
        find_documents_with_set_of_terms([["hydrogen", "heat"]], documents),
        {
            "['hydrogen', 'heat']": [False, True, False],
            "has_any_terms": [False, True, False],
        },
    )
    assert_equal(
        find_documents_with_set_of_terms(
            [["heat pump"], ["hydrogen", "heat"]], documents
        ),
        {
            "['heat pump']": [True, False, False],
            "['hydrogen', 'heat']": [False, True, False],
            "has_any_terms": [True, True, False],
        },
    )
    assert_equal(
        find_documents_with_set_of_terms(
            [["heat pump"], ["hydrogen", "heat"]], documents
        ),
        {
            "['heat pump']": [True, False, False],
            "['hydrogen', 'heat']": [False, True, False],
            "has_any_terms": [True, True, False],
        },
    )


def test_find_matches():
    corpus = {"id1": "best heat pump", "id2": "hydrogen heat"}
    Query = QueryTerms(corpus)
    assert Query.text_corpus == ["best heat pump", "hydrogen heat"]
    assert Query.document_ids == ["id1", "id2"]

    # Note that tokenised texts are preprocessed in a specific way
    corpus_tokenised = {"id1": ["best", "heat_pump"], "id2": ["hydrogen", "heat"]}
    QueryTokenised = QueryTerms(corpus_tokenised)
    assert QueryTokenised.text_corpus == [" best heat pump ", " hydrogen heat "]
    assert QueryTokenised.document_ids == ["id1", "id2"]

    search_terms = [["heat pump"]]
    expected_output = pd.DataFrame(
        {
            "['heat pump']": [True, False],
            "has_any_terms": [True, False],
            "id": ["id1", "id2"],
        }
    )
    assert Query.find_matches(search_terms).equals(expected_output)
    assert QueryTokenised.find_matches(search_terms).equals(expected_output)

    search_terms = [["heat pump"], ["hydrogen"]]
    expected_output = pd.DataFrame(
        {
            "['heat pump']": [True, False],
            "['hydrogen']": [False, True],
            "has_any_terms": [True, True],
            "id": ["id1", "id2"],
        }
    )
    assert Query.find_matches(search_terms).equals(expected_output)
    assert QueryTokenised.find_matches(search_terms).equals(expected_output)
