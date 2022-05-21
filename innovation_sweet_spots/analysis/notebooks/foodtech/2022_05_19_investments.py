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
import innovation_sweet_spots.analysis.wrangling_utils as wu
import importlib

importlib.reload(wu)

COLUMN_CATEGORIES = wu.dealroom.COLUMN_CATEGORIES

# %%
# Initialise a Dealroom wrangler instance
DR = wu.DealroomWrangler()

# %%
# Number of companies
len(DR.company_data)

# %%
# Number of funding rounds
len(DR.funding_rounds)

# %%
# Currencies that are not covered by our conversion approach
Converter = wu.CurrencyConverter()
COLUMN = "EACH ROUND CURRENCY"
all_dealroom_currencies = set(DR.explode_dealroom_table(COLUMN)[COLUMN].unique())
all_dealroom_currencies.remove("n/a")
all_dealroom_currencies.difference(Converter.currencies)

# %%
subindustry_counts = (
    DR.company_subindustries.groupby("SUB INDUSTRIES")
    .agg(counts=("id", "count"))
    .sort_values("counts", ascending=False)
)

# %%
subindustry_counts.head(20)


# %%
tag_counts = (
    DR.company_tags.groupby("TAGS")
    .agg(counts=("id", "count"))
    .sort_values("counts", ascending=False)
)

# %%
tag_counts.head(20)

# %%
DR.explode_dealroom_table("REVENUE MODEL").groupby("REVENUE MODEL").count()

# %%
DR.explode_dealroom_table("B2B/B2C").groupby("B2B/B2C").count()

# %%
subindustry_counts.head(20).index

# %%
SUBINDUSTRIES = [
    "innovative food",
    "food logistics & delivery",
    "agritech",
    "in-store retail & restaurant tech",
    "kitchen & cooking tech",
    "biotechnology",
    "waste solution",
    "content production",
    "social media",
    "pharmaceutical",
    "health platform",
]

# %%
ind = SUBINDUSTRIES[0]
ids_in_industry = DR.company_subindustries.query(
    "`SUB INDUSTRIES` == @ind"
).id.to_list()
deals = DR.funding_rounds.query("id in @ids_in_industry")

# %%
deals["EACH ROUND CURRENCY"].unique()

# %%
DR.company_data["TOTAL FUNDING (EUR M)"]

# %%
import importlib

importlib.reload(wu)

# %%
df = wu.convert_currency(
    funding=(
        deals.query("`EACH ROUND AMOUNT` != 'n/a'").query(
            "`EACH ROUND CURRENCY` != 'n/a'"
        )
    ),
    date_column="EACH ROUND DATE",
    amount_column="EACH ROUND AMOUNT",
    currency_column="EACH ROUND CURRENCY",
    target_currency="GBP",
)

# %%
df

# %%
