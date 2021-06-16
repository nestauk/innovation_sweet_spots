"""
Guardian specific utils

"""
from innovation_sweet_spots import PROJECT_DIR
import requests
import json
import os
import dotenv

dotenv.load_dotenv(PROJECT_DIR)


def get_guardian_api_key():
    api_key = open(os.environ["GUARDIAN_API_KEY"], "r").read()
    return api_key


"""
TO DO:
- Formulate search query
- Process results (check how many pages, go through all pages)
- Save the results, and the query
- If the same query already exists, get the results (not to use up the API limits)
- Perhaps, keep all queries in a lookup table?

In addition, separate routine for
- Find sentence where the term is mentioned
- Check sentiment
- Find typical words for good and bad sentiment
- Maybe check also how others have done this

+ Number of articles over time
+ Sentiment over time

"""

# set up base url
base_url = "https://content.guardianapis.com/"

# set up parameters
search_keyword = "Brexit OR (Theresa AND May)"
data_format = "json"
section = "politics"
from_date = "2007-01-01"
to_date = "2021-06-01"
page_size = 100
order_by = "newest"
production_office = "uk"
lang = "en"

finalized_url = "{}search?/q={}&format={}&section={}&from-date={}&to-date={}&page={}&page-size={}&order-by={}&production-office={}&lang={}&api-key={}".format(
    base_url,
    search_keyword,
    data_format,
    section,
    from_date,
    to_date,
    page,
    page_size,
    order_by,
    production_office,
    lang,
    api_key,
)
