"""
Utils for calling Guardian news API

Note the rate limits which, to the best of our knowledge, are 12 calls per second
and 5000 calls in total per day
"""
from innovation_sweet_spots import PROJECT_DIR, config, logging
from innovation_sweet_spots.getters.path_utils import GUARDIAN_PATH
import requests
import json
import os
import dotenv

dotenv.load_dotenv(PROJECT_DIR)
dotenv.load_dotenv(PROJECT_DIR / ".env")

from urllib.parse import urlencode, quote
import requests
import time

BASE_URL = "https://content.guardianapis.com/"
API_KEY = open(os.environ["GUARDIAN_API_KEY"], "r").read()

API_RESULTS_DIR = GUARDIAN_PATH / "api_results"
API_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_url(
    search_term: str, api_key: str = API_KEY, adjusted_parameters: dict = {}
):
    """NB: Use double quotes for the search term"""
    parameters = config["guardian_api"].copy()
    parameters["api-key"] = api_key
    for key in adjusted_parameters:
        parameters[key] = adjusted_parameters[key]
    search_query = f'q="{quote(search_term)}"&'
    url = f"{BASE_URL}search?" + search_query + urlencode(parameters)
    return url


def get_request(url):
    """Return response"""
    r = requests.get(url)
    time.sleep(0.25)
    return r


def get_cache_filename(search_term, fpath=API_RESULTS_DIR):
    return f"{fpath / quote(search_term)}.json"


def get_content_from_cache(search_term, fpath=API_RESULTS_DIR):
    """Very basic caching"""
    cached_dict = get_cache_filename(search_term, fpath)
    if os.path.exists(cached_dict):
        with open(cached_dict, "r") as infile:
            results = json.load(infile)
        logging.info(f"Loading results from {cached_dict}")
        return results
    else:
        return False


def save_content_to_cache(search_term, results, fpath=API_RESULTS_DIR):
    """Very basic caching"""
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
):
    # Check if we have already made such a search
    if use_cached:
        results_list = get_content_from_cache(search_term, fpath)
        if results_list is not False:
            return results_list

    # Do a new search
    url = create_url(search_term, api_key)
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
                url = create_url(search_term, api_key, {"page": current_page})
                r = get_request(url)
                if r.status_code == 200:
                    results_list.append(r.json()["response"]["results"])
        results_list = [r for page_results in results_list for r in page_results]
        percentage_collected = round(len(results_list) / n_total_results * 100)
        # Save the results
        logging.info(f"Collected {len(results_list)} ({percentage_collected}%) results")
        if save_to_cache:
            save_content_to_cache(search_term, results_list, fpath)
        return results_list
