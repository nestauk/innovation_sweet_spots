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
import missingno as msno

importlib.reload(wu)

# %%
# Initialise a Dealroom wrangler instance
DR = wu.DealroomWrangler()

# %%
# Check the number of companies
len(DR.company_data)

# %% [markdown]
# ### Industries and sub-industries

# %%
DR.company_subindustries

# %% [markdown]
# ## Making sense of columns and missing values
#
# Basic descriptors:
# - 'id',
# - 'NAME',
# - 'PROFILE URL',
# - 'WEBSITE',
# - 'TAGLINE',
# - 'LISTS',
# - 'COMPANY STATUS',
# - 'LOGO',
# - 'PIC NUMBER'
# - 'TRADE REGISTER NUMBER',
#
# Labels
# - 'TAGS',
# - 'B2B/B2C',
# - 'REVENUE MODEL',
# - 'INDUSTRIES',
# - 'SUB INDUSTRIES',
# - 'DELIVERY METHOD',
# - 'TECHNOLOGIES',
# - 'SDGS',
# - 'TECH STACK DATA (BY PREDICTLEADS)',
#
# Timelines
# - 'LAUNCH DATE',
# - 'CLOSING DATE',
# - 'YEAR COMPANY BECAME UNICORN',
# - 'YEAR COMPANY BECAME FUTURE UNICORN',
#
# Location
# - 'ADDRESS',
# - 'HQ REGION',
# - 'HQ COUNTRY',
# - 'HQ CITY',
# - 'LATITUDE',
# - 'LONGITUDE',
# - 'LOCATIONS',
# - 'CUSTOM HQ REGIONS',
# - 'FOUNDING LOCATION',
#
# Team
# - 'TEAM (DEALROOM)',
# - 'TEAM (EDITORIAL)',
# - 'FOUNDERS',
# - 'FOUNDERS STATUSES',
# - 'FOUNDERS GENDERS',
# - 'FOUNDERS IS SERIAL',
# - 'FOUNDERS BACKGROUNDS',
# - 'FOUNDERS UNIVERSITIES',
# - 'FOUNDERS COMPANY EXPERIENCE',
# - 'FOUNDERS FIRST DEGREE',
# - 'FOUNDERS FIRST DEGREE YEAR',
# - 'FOUNDERS LINKEDIN',
# - 'FOUNDERS FOUNDED COMPANIES TOTAL FUNDING',
# - 'NUMBER OF ALUMNI EUROPEAN FOUNDERS THAT RAISED > 10M',
#
# Workforce
# - 'EMPLOYEES',
# - 'EMPLOYEES (2016,2017,2018,2019,2020,2021,2022)',
# - 'EMPLOYEES IN HQ country (2016,2017,2018,2019,2020,2021,2022)',
# - 'EMPLOYEE RANK 3/6/12 MONTHS',
#
# Investors
# - 'INVESTORS',
# - 'EACH INVESTOR TYPES',
# - 'LEAD INVESTORS',
#
# Funding
# - 'TOTAL FUNDING (EUR M)',
# - 'TOTAL FUNDING (USD M)',
# - 'LAST ROUND',
# - 'LAST FUNDING',
# - 'LAST FUNDING DATE',
# - 'FIRST FUNDING DATE',
# - 'SEED YEAR',
# - 'OWNERSHIPS',
# - 'GROWTH STAGE',
# - 'YEARLY GROWTH (SIMILARWEB)',
# - 'ALEXA GROWTH (ALL TIME)',
# - 'INCOME STREAMS',
# - 'CORE SIDE VALUE',
#
# Funding (detailed)
# - 'EACH ROUND TYPE',
# - 'EACH ROUND AMOUNT',
# - 'EACH ROUND CURRENCY',
# - 'EACH ROUND DATE',
# - 'TOTAL ROUNDS NUMBER',
# - 'EACH ROUND INVESTORS',
#
# Financials
# - 'LAST KPI DATE',
# - 'PROFIT (2016,2017,2018,2019,2020,2021,2022,2023)',
# - 'PROFIT MARGIN (2015,2016,2017,2018,2019,2020,2021,2022,2023)',
# - 'EBITDA (2016,2017,2018,2019,2020,2021,2022,2023)',
# - 'EBITDA MARGIN (2015,2016,2017,2018,2019,2020,2021,2022,2023)',
# - 'REVENUE (2016,2017,2018,2019,2020,2021,2022,2023)',
# - 'REVENUE GROWTH (2015,2016,2017,2018,2019,2020,2021,2022,2023)',
# - 'R&D MARGIN (2015,2016,2017,2018,2019,2020,2021,2022,2023)',
# - 'FINANCIALS CURRENCY',
# - 'VALUATION',
# - 'VALUATION CURRENCY',
# - 'VALUATION (EUR)',
# - 'VALUATION (USD)',
# - 'VALUATION DATE',
# - 'HISTORICAL VALUATIONS - DATES',
# - 'HISTORICAL VALUATIONS - VALUES EUR M',
# - 'HISTORICAL VALUATIONS - VALUES USD M',
# - 'TRADING MULTIPLES EV/REVENUE (2017,2018,2019,2020,2021,2022,2023)',
# - 'TRADING MULTIPLES EV/EBITDA (2017,2018,2019,2020,2021,2022,2023)',
#
# Social and web
# - 'FACEBOOK LIKES',
# - 'TWITTER FOLLOWERS',
# - 'TWITTER TWEETS',
# - 'TWITTER FAVORITES',
# - 'SW TRAFFIC 6 MONTHS',
# - 'SW TRAFFIC 12 MONTHS',
# - 'ANGELLIST',
# - 'FACEBOOK',
# - 'TWITTER',
# - 'LINKEDIN',
# - 'GOOGLE PLAY LINK',
# - 'ITUNES LINK',
# - 'APP DOWNLOADS LATEST (IOS)',
# - 'APP DOWNLOADS 6 MONTHS (IOS)',
# - 'APP DOWNLOADS 12 MONTHS (IOS)',
# - 'APP DOWNLOADS LATEST (ANDROID)',
# - 'APP DOWNLOADS 6 MONTHS (ANDROID)',
# - 'APP DOWNLOADS 12 MONTHS (ANDROID)',
# - 'TRAFFIC COUNTRIES',
# - 'TRAFFIC SOURCES',
# - 'SIMILARWEB RANK 3/6/12 MONTHS',
# - 'APP RANK 3/6/12 MONTHS',
#

# %%
COLUMN_CATEGORIES = {
    "Basic descriptors": [
        "id",
        "NAME",
        "PROFILE URL",
        "WEBSITE",
        "TAGLINE",
        "LISTS",
        "COMPANY STATUS",
        "LOGO",
        "PIC NUMBER",
        "TRADE REGISTER NUMBER",
    ],
    "Labels": [
        "TAGS",
        "B2B/B2C",
        "REVENUE MODEL",
        "INDUSTRIES",
        "SUB INDUSTRIES",
        "DELIVERY METHOD",
        "TECHNOLOGIES",
        "SDGS",
        "TECH STACK DATA (BY PREDICTLEADS)",
    ],
    "Timelines": [
        "LAUNCH DATE",
        "CLOSING DATE",
        "YEAR COMPANY BECAME UNICORN",
        "YEAR COMPANY BECAME FUTURE UNICORN",
    ],
    "Location": [
        "ADDRESS",
        "HQ REGION",
        "HQ COUNTRY",
        "HQ CITY",
        "LATITUDE",
        "LONGITUDE",
        "LOCATIONS",
        "CUSTOM HQ REGIONS",
        "FOUNDING LOCATION",
    ],
    "Team": [
        "TEAM (DEALROOM)",
        "TEAM (EDITORIAL)",
        "FOUNDERS",
        "FOUNDERS STATUSES",
        "FOUNDERS GENDERS",
        "FOUNDERS IS SERIAL",
        "FOUNDERS BACKGROUNDS",
        "FOUNDERS UNIVERSITIES",
        "FOUNDERS COMPANY EXPERIENCE",
        "FOUNDERS FIRST DEGREE",
        "FOUNDERS FIRST DEGREE YEAR",
        "FOUNDERS LINKEDIN",
        "FOUNDERS FOUNDED COMPANIES TOTAL FUNDING",
        "NUMBER OF ALUMNI EUROPEAN FOUNDERS THAT RAISED > 10M",
    ],
    "Workforce": [
        "EMPLOYEES",
        "EMPLOYEES (2016,2017,2018,2019,2020,2021,2022)",
        "EMPLOYEES IN HQ country (2016,2017,2018,2019,2020,2021,2022)",
        "EMPLOYEE RANK 3/6/12 MONTHS",
    ],
    "Investors": [
        "INVESTORS",
        "EACH INVESTOR TYPES",
        "LEAD INVESTORS",
    ],
    "Funding": [
        "TOTAL FUNDING (EUR M)",
        "TOTAL FUNDING (USD M)",
        "LAST ROUND",
        "LAST FUNDING",
        "LAST FUNDING DATE",
        "FIRST FUNDING DATE",
        "SEED YEAR",
        "OWNERSHIPS",
        "GROWTH STAGE",
        "YEARLY GROWTH (SIMILARWEB)",
        "ALEXA GROWTH (ALL TIME)",
        "INCOME STREAMS",
        "CORE SIDE VALUE",
        "TOTAL ROUNDS NUMBER",
    ],
    "Funding (detailed)": [
        "EACH ROUND TYPE",
        "EACH ROUND AMOUNT",
        "EACH ROUND CURRENCY",
        "EACH ROUND DATE",
        "EACH ROUND INVESTORS",
    ],
    "Financials": [
        "LAST KPI DATE",
        "PROFIT (2016,2017,2018,2019,2020,2021,2022,2023)",
        "PROFIT MARGIN (2015,2016,2017,2018,2019,2020,2021,2022,2023)",
        "EBITDA (2016,2017,2018,2019,2020,2021,2022,2023)",
        "EBITDA MARGIN (2015,2016,2017,2018,2019,2020,2021,2022,2023)",
        "REVENUE (2016,2017,2018,2019,2020,2021,2022,2023)",
        "REVENUE GROWTH (2015,2016,2017,2018,2019,2020,2021,2022,2023)",
        "R&D MARGIN (2015,2016,2017,2018,2019,2020,2021,2022,2023)",
        "FINANCIALS CURRENCY",
        "VALUATION",
        "VALUATION CURRENCY",
        "VALUATION (EUR)",
        "VALUATION (USD)",
        "VALUATION DATE",
        "HISTORICAL VALUATIONS - DATES",
        "HISTORICAL VALUATIONS - VALUES EUR M",
        "HISTORICAL VALUATIONS - VALUES USD M",
        "TRADING MULTIPLES EV/REVENUE (2017,2018,2019,2020,2021,2022,2023)",
        "TRADING MULTIPLES EV/EBITDA (2017,2018,2019,2020,2021,2022,2023)",
    ],
    "Social and web": [
        "FACEBOOK LIKES",
        "TWITTER FOLLOWERS",
        "TWITTER TWEETS",
        "TWITTER FAVORITES",
        "SW TRAFFIC 6 MONTHS",
        "SW TRAFFIC 12 MONTHS",
        "ANGELLIST",
        "FACEBOOK",
        "TWITTER",
        "LINKEDIN",
        "GOOGLE PLAY LINK",
        "ITUNES LINK",
        "APP DOWNLOADS LATEST (IOS)",
        "APP DOWNLOADS 6 MONTHS (IOS)",
        "APP DOWNLOADS 12 MONTHS (IOS)",
        "APP DOWNLOADS LATEST (ANDROID)",
        "APP DOWNLOADS 6 MONTHS (ANDROID)",
        "APP DOWNLOADS 12 MONTHS (ANDROID)",
        "TRAFFIC COUNTRIES",
        "TRAFFIC SOURCES",
        "SIMILARWEB RANK 3/6/12 MONTHS",
        "APP RANK 3/6/12 MONTHS",
    ],
}


# %%
list(COLUMN_CATEGORIES.keys())

# %%
msno.bar(DR.company_data[COLUMN_CATEGORIES["Basic descriptors"]])

# %%
msno.bar(DR.company_data[COLUMN_CATEGORIES["Labels"]])

# %%
msno.bar(DR.company_data[COLUMN_CATEGORIES["Location"]])

# %%
msno.bar(DR.company_data[COLUMN_CATEGORIES["Funding"]])

# %%
msno.bar(DR.company_data[COLUMN_CATEGORIES["Funding (detailed)"]])

# %%
msno.bar(DR.company_data[COLUMN_CATEGORIES["Investors"]])

# %%
msno.bar(DR.company_data[COLUMN_CATEGORIES["Financials"]])

# %%
msno.bar(DR.company_data[COLUMN_CATEGORIES["Social and web"]])

# %%
msno.bar(DR.company_data[COLUMN_CATEGORIES["Team"]])

# %%
msno.bar(DR.company_data[COLUMN_CATEGORIES["Workforce"]])

# %% [markdown]
# ### Funding time series

# %%
col_name = "PROFIT (2016,2017,2018,2019,2020,2021,2022,2023)"

# %%
df = DR.explode_timeseries(col_name)
# df.query("PROFIT != 'n/a'")

# %%
df

# %%
cols = [
    "EACH ROUND TYPE",
    "EACH ROUND AMOUNT",
    "EACH ROUND CURRENCY",
    "EACH ROUND DATE",
]

# %%
df = (
    DR.company_data[["id"] + COLUMN_CATEGORIES["Funding (detailed)"]].fillna("n/a")
).copy()

# %%
for col in COLUMN_CATEGORIES["Funding (detailed)"]:
    df[col] = df[col].apply(lambda x: wu.split_comma_seperated_string(x, ";"))
df_ = df.copy()

# %%
df_.explode(cols)

# %%
for col in COLUMN_CATEGORIES["Funding (detailed)"]:
    df[col] = df[col].apply(lambda x: len(x))
df.drop("id", axis=1)

# %%
df[df["EACH ROUND TYPE"] != df["EACH ROUND INVESTORS"]]

# %%
import pandas as pd

pd.set_option("max_colwidth", 200)
df_[df["EACH ROUND TYPE"] != df["EACH ROUND INVESTORS"]]

# %%
DR.company_data["EACH ROUND INVESTORS"].loc[12912]

# %%

# %%
df.explode(COLUMN_CATEGORIES["Funding (detailed)"])

# %%
DR.company_data

# %%
importlib.reload(wu)
DR = wu.DealroomWrangler()

# %%

# %%

# %%
df.explode([col_name, "year"])

# %%

# %%
DR.company_data

# %%
DR.explode_dealroom_table("PROFIT (2016,2017,2018,2019,2020,2021,2022,2023)")


# %%

# %% [markdown]
# ### Company descriptions

# %%
# Check the lenghts of company descriptions/taglines
def text_word_count(text: str) -> int:
    """Calculates word count of the input text"""
    if type(text) is str:
        return len(text.split(" "))
    else:
        return 0


# %% [markdown]
# ### Investment data

# %%
word_counts = DR.company_data.TAGLINE.apply(text_word_count)
word_counts.plot.hist()

# %%
# Median word count
print(word_counts.median())

# %%
# Number/fraction of companies without a tagline
(word_counts == 0).sum(), round((word_counts == 0).sum() / len(word_counts), 3)

# %%
