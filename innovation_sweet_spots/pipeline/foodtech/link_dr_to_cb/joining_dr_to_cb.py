# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from innovation_sweet_spots.getters.dealroom import get_foodtech_companies
from innovation_sweet_spots.getters.crunchbase import get_crunchbase_orgs
from tempfile import TemporaryDirectory
from jacc_hammer.fuzzy_hash import Cos_config, Fuzzy_config, match_names_stream
from pathlib import Path
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.pipeline.foodtech.link_dr_to_cb.utils import (
    CB_COLS_RENAME,
    DR_COLS_RENAME,
    EVALUATION_COLS,
    DR_URL_COLS,
    CB_URL_COLS,
    cols_replace_space_and_lowercase,
    find_dr_cb_matches,
    find_url_matches,
    make_combined_dr_cb_lookup,
    add_clean_name_col,
    update_cb_countries_to_match_dr,
    update_dr_countries_to_match_cb,
)
import numpy as np
import pandas as pd

pd.set_option("mode.chained_assignment", None)

# %% [markdown]
# # Matching Dealroom companies to Crunchbase companies

# %%
# Load datasets, add a cleaned company name column and standardise country location names
dr = (
    get_foodtech_companies()
    .pipe(cols_replace_space_and_lowercase)
    .drop_duplicates()
    .dropna(subset=["id"])
    .pipe(add_clean_name_col)
    .pipe(update_dr_countries_to_match_cb)
)
cb = (
    get_crunchbase_orgs().pipe(add_clean_name_col).pipe(update_cb_countries_to_match_dr)
)

# %% [markdown]
# ## Exact Matches
# First start by finding exact matches using related urls and company name combined with country.

# %%
# Find dealroom to crunchbase matches using urls
combined_url_matches = []
dr_companies_left_to_match = dr.copy()
for dr_url, cb_url in zip(DR_URL_COLS, CB_URL_COLS):
    url_matches = find_url_matches(dr_companies_left_to_match, cb, dr_url, cb_url)
    # Add current matches to the combined dr to cb lookup
    combined_url_matches.append(url_matches)
    combined_dr_cb_lookup = make_combined_dr_cb_lookup(combined_url_matches)
    # Update list of dr ids that have been matched to cb ids
    dr_matched_ids = list(combined_dr_cb_lookup.id_dr.values)
    # Update dataframe of dealroom companies that have not been matched yet
    dr_companies_left_to_match = dr.query(f"id not in {dr_matched_ids}")

# %%
combined_dr_cb_lookup

# %%
dr_companies_left_to_match

# %%
# Find exact matches of companies with the same country and name
matches_from_country_and_name = dr_companies_left_to_match.dropna(
    subset=["clean_name"]
).pipe(
    find_dr_cb_matches,
    cb_companies=cb,
    dr_on=["clean_name", "hq_country"],
    cb_on=["clean_name", "country"],
)
# Add exact matches of same country and name to the combined dr to cb lookup
combined_dr_cb_lookup = make_combined_dr_cb_lookup(
    [combined_dr_cb_lookup, matches_from_country_and_name]
)
# Update list of dr ids that have been matched to cb ids
dr_matched_ids = list(combined_dr_cb_lookup.id_dr.values)
# Update dataframe of dealroom companies that have not been matched yet
dr_companies_left_to_match = dr.query(f"id not in {dr_matched_ids}")

# %%
combined_dr_cb_lookup

# %%
18743 - 18309

# %%
dr_companies_left_to_match

# %% [markdown]
# ## Fuzzy Matches
# Then use fuzzy matching for each country to find additional matches which can then be evaluated to check how good the match is.<br>
# <br>
# Running the cell below will produce csvs containing fuzzy matches for each country in `outputs/dr_to_cb_fuzzy_matches`.<br>

# %%
# Load configs
cos_config = Cos_config()
fuzzy_config = Fuzzy_config()
# Save fuzzy matches to dir
DR_TO_CB_FUZZY_MATCHES_DIR = PROJECT_DIR / "outputs/dr_to_cb_fuzzy_matches/"
DR_TO_CB_FUZZY_MATCHES_DIR.mkdir(exist_ok=True)
# Create list of countries in dealroom dataset
dr_countries = [
    country
    for country in dr_companies_left_to_match.hq_country.unique()
    if pd.isnull(country) is False
]
# Set settings
sim_mean_min = 70
chunksize = 100_000
for country in dr_countries:
    # Create temp directory
    tmp_dir = Path(TemporaryDirectory().name)
    tmp_dir.mkdir()
    dr_country_subset = dr_companies_left_to_match.query(
        f"hq_country == '{country}'"
    ).reset_index(drop=True)
    cb_country_subset = cb.query(f"country == '{country}'").reset_index(drop=True)
    if len(cb_country_subset) != 0:
        # Do fuzzy matching
        cb_names = cb_country_subset.clean_name.to_list()
        dr_names = dr_country_subset.clean_name.to_list()
        fuzzy_name_matches = pd.concat(
            match_names_stream(
                [dr_names, cb_names],
                chunksize=chunksize,
                tmp_dir=tmp_dir,
                cos_config=cos_config,
                fuzzy_config=fuzzy_config,
            )
            # Filter for matches over sim mean min threshold
        ).query(f"sim_mean >= {sim_mean_min}")
        # Add extra information that can be used in the evaluation step
        fuzzy_name_matches_with_info = (
            fuzzy_name_matches.merge(
                right=cb_country_subset, right_index=True, left_on="y", how="left"
            )
            .rename(columns=CB_COLS_RENAME)
            .merge(right=dr_country_subset, right_index=True, left_on="x", how="left")
            .rename(columns=DR_COLS_RENAME)
            .sort_values(by=["sim_mean"], ascending=False)
            .reset_index(drop=True)[EVALUATION_COLS]
            .assign(label_good_match=np.nan)
        )
        country_fuzzy_match_save_path = (
            DR_TO_CB_FUZZY_MATCHES_DIR
            / f"{country.lower().replace(' ', '_')}_fuzzy_name_matches_dr_to_cb.csv"
        )
        # Save a csv of fuzzy matches if there are some matches
        if len(fuzzy_name_matches_with_info) > 0:
            fuzzy_name_matches_with_info.to_csv(country_fuzzy_match_save_path)

# %% [markdown]
# ## Manual evaluation

# %% [markdown]
# Manually evaluate the matches in the csvs in `outputs/dr_to_cb_fuzzy_matches` by adding a `y` for a good match or `n` for a bad match in the `label_good_match` column for each csv.<br>
# <br>
# The next cells will add the `y` labelled companies to the `combined_dr_cb_lookup` variable and then save the combined matches.

# %%
DR_TO_CB_FUZZY_MATCHES_DIR = PROJECT_DIR / "outputs/dr_to_cb_fuzzy_matches/"

# %%
# Select dr_id and matching cb_id for companies marked as good matches in the csvs
yes_fuzzy_matches = (
    pd.concat(
        [
            pd.read_csv(file, index_col=0).query("label_good_match == 'y'")
            for file in DR_TO_CB_FUZZY_MATCHES_DIR.iterdir()
            if file.suffix == ".csv"
        ]
    )
    .reset_index(drop=True)[["dr_id", "cb_id"]]
    .rename(columns={"dr_id": "id_dr", "cb_id": "id_cb"})
)

# %%
# Add fuzzy matches to combined_dr_cb_lookup
combined_dr_cb_lookup = pd.concat(
    [combined_dr_cb_lookup, yes_fuzzy_matches]
).reset_index(drop=True)

# %%
# Save lookup file
DR_CB_LINK_DIR = PROJECT_DIR / "inputs/data/dr_cb_link/"
DR_CB_LINK_DIR.mkdir(exist_ok=True)
combined_dr_cb_lookup.to_csv(DR_CB_LINK_DIR / "dr_cb_lookup.csv")

# %%
