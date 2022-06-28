from unittest import mock
from pathlib import Path
import json
import os

from innovation_sweet_spots.getters.guardian import (
    save_content_to_cache,
    get_content_from_cache,
    get_cache_filename,
    API_RESULTS_DIR,
    search_content,
)


def remove_file(fpath):
    if os.path.exists(fpath):
        os.remove(fpath)


def test_get_cache_filename():
    assert get_cache_filename("heat pump") == str(API_RESULTS_DIR / "heat%20pump.json")
    assert get_cache_filename("nesta") == str(API_RESULTS_DIR / "nesta.json")
    assert get_cache_filename("nesta", fpath=Path("")) == "nesta.json"


def test_save_content_to_cache():
    mock_results = [{"one": 1, "two": 2}]
    mock_query = "mock query"
    mock_output_file = "mock%20query.json"
    save_content_to_cache(mock_query, mock_results, fpath=Path(""))
    try:
        with open(mock_output_file, "r") as infile:
            r = json.load(infile)
        assert r == mock_results
        remove_file(mock_output_file)
    except AssertionError as error:
        remove_file(mock_output_file)
        raise error


def test_get_content_to_cache():
    mock_results = [{"one": 1, "two": 2}]
    mock_query = "mock query"
    mock_output_file = "mock%20query.json"
    with open(mock_output_file, "w") as outfile:
        json.dump(mock_results, outfile)
    r = get_content_from_cache(mock_query, fpath=Path(""))
    try:
        assert r == mock_results
        remove_file(mock_output_file)
    except AssertionError as error:
        remove_file(mock_output_file)
        raise error


def test_create_url():
    pass


@mock.patch("innovation_sweet_spots.getters.guardian.get_request")
def test_search_content(mock_get_request):
    mock_query = "mock query"
    mock_output_file = "mock%20query.json"

    # Set up a mock API get request response
    mock_results = [{"one": 1}, {"two": 2}]
    mock_response_json = {
        "results": mock_results,
        "total": 3,
        "pages": 1,
        "currentPage": 1,
    }
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json = lambda: {"response": mock_response_json}
    mock_get_request.return_value = mock_response

    # Test the case when searching for a new term
    results = search_content(mock_query, api_key=None, save_to_cache=False)
    assert results == mock_results

    # Test the case when caching the results
    results = search_content(
        mock_query, api_key=None, save_to_cache=True, fpath=Path("")
    )
    print(results)
    try:
        assert results == mock_results
        with open(mock_output_file, "r") as infile:
            r = json.load(infile)
        assert r == mock_results
        remove_file(mock_output_file)
    except AssertionError as error:
        remove_file(mock_output_file)
        raise error

    # Test the case when a result is already in the cache
    cached_results = [{"cached": 1, "result": 2}]
    with open(mock_output_file, "w") as outfile:
        json.dump(cached_results, outfile)
    results = search_content(mock_query, api_key=None, fpath=Path(""))
    try:
        assert results != mock_results
        assert results == cached_results
        remove_file(mock_output_file)
    except AssertionError as error:
        remove_file(mock_output_file)
        raise error

    # Test the case when the call fails
    mock_response.status_code = 404
    results = search_content(mock_query, api_key=None, fpath=Path(""))
    assert results.status_code == mock_response.status_code

    # Test the case when more than one page of results
    mock_response.status_code = 200
    mock_response_json["pages"] = 3
    results = search_content(mock_query, api_key=None, save_to_cache=False)
    assert results == (mock_results + mock_results + mock_results)
    # Test the case when more than one page of results, but just checking
    results = search_content(
        mock_query, api_key=None, save_to_cache=False, only_first_page=True
    )
    assert results == mock_results
