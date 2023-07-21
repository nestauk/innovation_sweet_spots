"""
Utils for calling Guardian news API

Note the rate limits which, to the best of our knowledge, are 1 call per second
and 500 calls in total per day
"""
from innovation_sweet_spots import PROJECT_DIR, config, logging
from innovation_sweet_spots.getters.path_utils import GUARDIAN_PATH
import requests
import json
import os
import dotenv
from urllib.parse import urlencode, quote
import requests
import time
from typing import List, Dict
from pathlib import Path

# Base url for calling the api
BASE_URL = "https://content.guardianapis.com/"
# Folders to store the api call results
API_RESULTS_DIR = GUARDIAN_PATH / "api_results"
API_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def setup_api_key(api_key: str = None, from_env_file: bool = True) -> str:
    """
    Initialises The Guardian api key

    Args:
        api_key (str, optional): The Guardian API key. Defaults to None.
        from_env_file (bool, optional): Whether to look for the API key in the .env file. Defaults to True.

    Returns:
        str: The Guardian API key. If it cannot find the API key in the .env file, it returns None.
    """
    if from_env_file:
        # If .env file exists
        if os.path.isfile(PROJECT_DIR / ".env"):
            # Load the .env file
            dotenv.load_dotenv(PROJECT_DIR / ".env")
            try:
                # Try loading API key
                return open(os.environ["GUARDIAN_API_KEY"], "r").read()
            except:
                # If the key is not in the .env file
                return None
    else:
        return api_key


# API key
API_KEY = setup_api_key()


def create_url(
    search_term: str, api_key: str = API_KEY, adjusted_parameters: dict = {}
) -> str:
    """
    Create the url for the Guardian API call

    Args:
        search_term (str): The search query
        api_key (str, optional): Guardian API key. Defaults to API_KEY.
        adjusted_parameters (dict, optional): Additional parameters to pass to the API. Defaults to {}.

    Returns:
        str: The url for the API call
    """
    # NB: Use double quotes for the search term
    parameters = config["guardian_api"].copy()
    parameters["api-key"] = api_key
    for key in adjusted_parameters:
        parameters[key] = adjusted_parameters[key]
    # Split the search terms on commmas and add ANDs between them
    if len(search_term.split(",")) <= 1:
        search_query = f'q="{quote(search_term)}"&'
    else:
        search_terms = search_term.split(",")
        search_query = f'q="{quote(search_terms[0])}"&'
        for i in search_terms[1:]:
            search_query = search_query.replace("&", f' AND "{quote(i)}"&')

    url = f"{BASE_URL}search?" + search_query + urlencode(parameters)
    return url


def get_request(url: str):
    """Return response"""
    r = requests.get(url)
    time.sleep(1)
    return r


def get_cache_filename(search_term: str, fpath: Path = API_RESULTS_DIR) -> str:
    """Generates the filename for the cached results"""
    return f"{fpath / quote(search_term)}.json"


def get_content_from_cache(search_term: str, fpath=API_RESULTS_DIR) -> List[Dict]:
    """
    Basic caching by fetching the results from a local json file

    Args:
        search_term (str): The search query
        fpath (_type_, optional): Where to find the cached results. Defaults to API_RESULTS_DIR.

    Returns:
        List[Dict]: List of articles, where each article is a dictionary
    """
    cached_dict = get_cache_filename(search_term, fpath)
    if os.path.exists(cached_dict):
        with open(cached_dict, "r") as infile:
            results = json.load(infile)
        logging.info(f"Loading results from {cached_dict}")
        return results
    else:
        return False


def save_content_to_cache(
    search_term: str, results, fpath: Path = API_RESULTS_DIR
) -> None:
    """
    Basic caching by saving the results to a local json file

    Args:
        search_term (str): The search query
        results (List[Dict]): List of articles, where each article is a dictionary
        fpath (_type_, optional): Where to save the results. Defaults to API_RESULTS_DIR.

    Returns:
        None
    """
    cached_dict = get_cache_filename(search_term, fpath)
    with open(cached_dict, "w") as outfile:
        json.dump(results, outfile)
    logging.info(f"Saved results in {cached_dict}")


def search_content(
    search_term: str,
    api_key: str = API_KEY,
    use_cached: bool = True,
    save_to_cache: bool = True,
    fpath=API_RESULTS_DIR,
    only_first_page: bool = False,
    adjusted_parameters: dict = {},
) -> List[Dict]:
    """
    Search the Guardian API for a given search term

    Args:
        search_term (str): The search query
        api_key (str, optional): Guardian API key. Defaults to API_KEY.
        use_cached (bool, optional): Whether to use cached results. Defaults to True.
        save_to_cache (bool, optional): Whether to save results to cache. Defaults to True.
        fpath (_type_, optional): Where to save the results. Defaults to API_RESULTS_DIR.
        only_first_page (bool, optional): Whether to only get the first page of results. Defaults to False.
        adjusted_parameters (dict, optional): Additional parameters to pass to the API. Defaults to {}.

    Returns:
        List[Dict]: List of articles, where each article is a dictionary
    """
    # Check if we have already made such a search
    if use_cached:
        results_list = get_content_from_cache(search_term, fpath)
        if results_list is not False:
            return results_list

    # Do a new search
    url = create_url(search_term, api_key, adjusted_parameters)
    r = get_request(url)
    if r.status_code != 200:
        return r
    else:
        response = r.json()["response"]
        n_total_results = response["total"]
        n_pages_total = response["pages"]
        current_page = response["currentPage"]  # should always = 1
        results_list = [response["results"]]
        # Get results from all pages
        if (n_pages_total > 1) and (not only_first_page):
            while current_page < n_pages_total:
                # Update url and call again
                current_page += 1
                adjusted_parameters_ = adjusted_parameters.copy()
                adjusted_parameters_["page"] = current_page
                url = create_url(search_term, api_key, adjusted_parameters_)
                r = get_request(url)
                if r.status_code == 200:
                    results_list.append(r.json()["response"]["results"])
        results_list = [r for page_results in results_list for r in page_results]
        if n_total_results != 0:
            percentage_collected = round(len(results_list) / n_total_results * 100)
            # Save the results
            if save_to_cache:
                save_content_to_cache(search_term, results_list, fpath)
        else:
            percentage_collected = "n/a"
        logging.info(f"Collected {len(results_list)} ({percentage_collected}%) results")
        return results_list
