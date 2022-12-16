# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Generating baseline for Guardian articles

# %%
from innovation_sweet_spots.getters import guardian
import importlib

# %%
from urllib.parse import urlencode, quote
import requests
import pandas as pd

# %%
# Check sections

# %%
r = requests.get(
    "https://content.guardianapis.com/sections?production-office=uk&from-date=2000-01-01&to-date=2021-12-31&page-size=50&api-key=test"
)


# %%
# for i, s in enumerate(r.json()['response']['results']):
#     print(i, s['webTitle'])

# %%
# guardian.create_url('obesity', API_KEY)

# %%
guardian.config["guardian_api"]


# %%
def define_minimal_config_with_dates(
    from_date: str = "2000-01-01", to_date: str = "2025-12-31"
):
    return {
        "data-format": "json",
        "from-date": from_date,
        "to-date": to_date,
        "page-size": 2,
        "order-by": "newest",
        "lang": "en",
        # 'production-office': 'uk',
    }


def define_query(BASE_URL: str, search_query: str, parameters: dict):
    return f"{BASE_URL}search?" + search_query + urlencode(parameters)


def get_total_article_counts(
    search_query: str = "",
    start_year: int = 2000,
    end_year: int = 2022,
    api_key: str = API_KEY,
):
    """"""
    articles_per_year = {}
    for year in range(start_year, end_year + 1):
        parameters = define_minimal_config_with_dates(f"{year}-01-01", f"{year}-12-31")
        parameters["api-key"] = API_KEY
        query = define_query(BASE_URL, search_query, parameters)
        r = guardian.get_request(query).json()
        articles_per_year[year] = r["response"]["total"]
    return articles_per_year


# %%
total_article_counts = get_total_article_counts(2000, 2022, API_KEY)

# %%
total_article_counts_df = pd.DataFrame(
    data={"year": total_article_counts.keys(), "counts": total_article_counts.values()}
)

# %%
import altair as alt
from innovation_sweet_spots import PROJECT_DIR

# %%
(
    alt.Chart(total_article_counts_df)
    .mark_bar()
    .encode(
        x=alt.X("year:O"),
        y=alt.Y("counts:Q"),
    )
)

# %%
total_article_counts_df.to_csv(
    PROJECT_DIR / "outputs/foodtech/interim/public_discourse/guardian_baseline.csv",
    index=False,
)

# %%
