# -*- coding: utf-8 -*-
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

# %%
""" Script to generate location-based features """

# %%
from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler

# %%
import pandas as pd
import numpy as np
from innovation_sweet_spots import logging

# %%
CB = CrunchbaseWrangler()

# %%
uk_orgs = CB.cb_organisations.query("country == 'United Kingdom'")

# %%
location_columns = [
    "country_code",
    "region",
    "city",
    "address",
    "postal_code",
    "location_id",
    "country",
]

# %%
uk_org_locations = uk_orgs[["id", "name"] + location_columns]

# %%
uk_org_locations.iloc[0]

# %%
uk_orgs[["id", "name"] + location_columns].info()


# %%
def print_top_values(df: pd.DataFrame, col: str, top_n: int = 5) -> pd.DataFrame:
    print(
        (
            df.groupby(col)
            .agg(counts=(col, "count"))
            .sort_values("counts", ascending=False)
            .head(top_n)
        )
    )


# %%
for col in location_columns:
    print(f"{col}: {len(uk_orgs[col].unique())}")

# %%
for col in location_columns:
    print_top_values(uk_orgs, col)
    print("---")

# %%
from innovation_sweet_spots.utils import geo
import importlib

importlib.reload(geo)

# %% [markdown]
# # Step 1: Using postcode lookup

# %%
postcode_data = geo.get_nspl()[["pcd", "pcd2", "pcds", "lat", "long"]]


# %%
# Preprocess the postcodes
def remove_spaces(text: str) -> str:
    return "".join(text.split())


def preprocess_postcode(text: str) -> str:
    if type(text) is str:
        return remove_spaces(text).lower()
    else:
        return ""


for col in ["pcd", "pcd2", "pcds"]:
    # Remove spaces
    postcode_data[col] = postcode_data[col].apply(preprocess_postcode)

# %%
postcode_data.head(3)

# %%
# Unique postal codes to lat, long
unique_postal_codes = [
    preprocess_postcode(p)
    for p in uk_orgs[-uk_orgs.postal_code.isnull()].postal_code.unique()
]

# %%
# Get lat, long of the postal codes
codes_lat_long = pd.DataFrame(data={"postal_code": unique_postal_codes}).merge(
    postcode_data, left_on="postal_code", right_on="pcd", how="left"
)

# %%
matches = codes_lat_long[-codes_lat_long.lat.isnull()]
unmatched = codes_lat_long[codes_lat_long.lat.isnull()]

# %%
uk_org_locations = (
    uk_org_locations.assign(
        postal_code_preproc=lambda x: x.postal_code.apply(preprocess_postcode)
    )
    .merge(
        codes_lat_long[["postal_code", "lat", "long"]],
        left_on="postal_code_preproc",
        right_on="postal_code",
        how="left",
    )
    .drop(["postal_code_preproc", "postal_code_y"], axis=1)
    .rename(columns={"postal_code_x": "postal_code"})
)


# %%
uk_org_locations.info()

# %% [markdown]
# # Step 2: Geolocating addresses

# %% [markdown]
# ## Generic London address

# %%
# Geo-locating all companies with generic London address
london_lat_long = geo.geolocate_address("London, England, United Kingdom")

# %%
df = uk_org_locations[uk_org_locations.lat.isnull()]

# %%
london_companies = uk_org_locations[uk_org_locations.lat.isnull()][
    (df.region == "England") & (df.city == "London") & (df.address.isnull())
].id.to_list()

uk_org_locations.loc[
    uk_org_locations.id.isin(london_companies), "lat"
] = london_lat_long["lat"]
uk_org_locations.loc[
    uk_org_locations.id.isin(london_companies), "long"
] = london_lat_long["lng"]

# %%
len(uk_org_locations[uk_org_locations.lat.isnull()])

# %% [markdown]
# ## Non-generic address

# %%
df.head()

# %%
# Compile addresses
cols = [
    "address",
    "region",
    "city",
    "postal_code",
    "country",
]

df = uk_org_locations[uk_org_locations.lat.isnull()].copy()
# df['full_address'] = [', '.join(row[]) for i, row in df.iterrows()]
# if True: df = df.iloc[0:5]

# %%
len(df)

# %%
addresses = []
for i, row in df.iterrows():
    # Filter fields with null values and join up into an address string
    address = ", ".join([row[col] for col in row[cols].dropna().index])
    addresses.append(address)
df["full_address"] = addresses

# %%
len(df["full_address"])

# %%
len(df["full_address"].unique())

# %%
# Get all addresses (to do: only the unique addresses)
bing_addresses = [geo.geolocate_address(a) for a in df["full_address"].to_list()[0:5]]

# %%
# To Do > turn into a script and save the outputs

# %% [markdown]
# # Step 3: Link lat, long to NUTS

# %%
from nuts_finder import NutsFinder

nf = NutsFinder(
    year=2021
)  # <-- expect a little bit of loading time here whilst it downloads some shapefiles
# nf.find(lat=53.406115, lon=-2.965604)  # <-- pretty quick

# %%
# Do this for several years, eg 2013, 2016, 2021

# %%
uk_org_locations.iloc[0:5]

# %%
# Get unique lat, long values
all_nuts = [
    nf.find(lat=row["lat"], lon=row["long"])
    for i, row in uk_org_locations.iloc[0:5].iterrows()
]


# %%
# all_nuts

# %% [markdown]
# # Step 4: Extract features from the BEIS indicators

# %%
