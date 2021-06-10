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
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
"""
Notebook for creating 'mock' outputs with examples of the type of insights we could obtain

"""

# %% [markdown]
# # Notebook for creating mock outputs

# %%
from innovation_sweet_spots.getters import gtr, crunchbase
import pandas as pd
import time
import numpy as np


# %%
def timeit(method):
    """
    Ref: https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
    """

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts) * 1000)
        else:
            print("%r  %2.2f ms" % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def create_documents(lists_of_texts):
    """
    Create documents from lists of texts for further analysis, e.g. to
    calculate tf-idf scores of n-grams. For example:
        (['one','two'], ['cat', 'dogs']) -> ['one cat', 'two dogs']
    Args:
        lists_of_skill_texts (iterable of list of str): Contains lists of text
            features (e.g. label or description) to be joined up and processed to
            create the "documents"; i-th element of each list corresponds
            to the i-th entity/document
    Yields:
        (str): Created documents
    """
    # Check if all lists have the same length
    if len({len(i) for i in lists_of_texts}) == 1:
        # Transpose the lists of skill texts
        transposed_lists_of_texts = map(list, zip(*lists_of_texts))
        # Join up the skill texts for each skills entity
        return (
            " ".join(document_texts) for document_texts in transposed_lists_of_texts
        )
    else:
        raise ValueError("All lists in lists_of_texts should have the same length")


def create_documents_from_dataframe(df, columns):
    """Build documents from texts in the table columns"""
    # Select columns to include in the document
    df_ = df[columns].fillna("").copy()
    # Preprocess project text
    text_lists = [df_[col].str.lower().to_list() for col in columns]
    # Create project documents
    docs = list(create_documents(text_lists))
    return docs


# %% [markdown]
# ## 1. Setup the analysis

# %% [markdown]
# ### Load in the data

# %%
from innovation_sweet_spots.getters import inputs

importlib.reload(gtr)

# %%
from innovation_sweet_spots.getters.inputs import GTR_PATH

# %%
# %%capture
# Crunchbase
cb_df = crunchbase.get_crunchbase_orgs()
# GTR
gtr_projects = gtr.get_gtr_projects()
gtr_funds = gtr.get_gtr_funds()
gtr_organisations = gtr.get_gtr_organisations()

# %%
gtr_projects.info()

# %% [markdown]
# ### Docs

# %%
cb_columns = ["name", "short_description", "long_description"]
cb_docs = create_documents_from_dataframe(cb_df, cb_columns)

# %%
gtr_columns = ["title", "abstractText", "techAbstractText"]
gtr_docs = create_documents_from_dataframe(gtr_projects, gtr_columns)

# %% [markdown]
# ### GTR link table

# %%
# Check GTR link table
pd.read_csv(GTR_PATH / "gtr_link_table.csv", nrows=5)

# %%
# # Collect all links between projects and funds (rel=='FUND')
# chunksize = 10000
# offset = -chunksize
# links_funds = pd.DataFrame()
# link_table_path = GTR_PATH / 'gtr_link_table.csv'
# df = pd.read_csv(link_table_path, nrows=chunksize, skiprows=range(1, chunksize+offset))
# unique_rels = set()
# while len(df) == chunksize:
#     df = pd.read_csv(link_table_path, nrows=chunksize, skiprows=range(1, chunksize+offset))
#     links_funds = links_funds.append(df[df.rel=='FUND'], ignore_index=True)

#     unique_rels = unique_rels.union(set(list(zip(df.rel.to_list(),df.table_name.to_list()))))
#     offset += chunksize

# %%
set(zip([r[1] for r in list(unique_rels)], [r[0] for r in list(unique_rels)]))

# %%
offset / 1e6

# %%
# links_funds.to_csv(GTR_PATH / 'gtr_link_projects_funds.csv', index=False)

# %%

# %%
len(links_funds)

# %% [markdown]
# ### Link funds to projects

# %%
from innovation_sweet_spots import PROJECT_DIR

OUTPUTS_DIR = PROJECT_DIR / "outputs/data"

# %%
links_funds = pd.read_csv(OUTPUTS_DIR / "gtr/link_projects_funds.csv")

# %%
gtr_funded_projects = gtr_funds.merge(links_funds)

# %%
gtr_funded_projects.groupby("category").agg({"id": "count"})

# %%
gtr_funded_projects.info()

# %%
gtr_funded_projects = (
    gtr_funded_projects[gtr_funded_projects.category == "INCOME_ACTUAL"]
    .sort_values("amount", ascending=False)
    .drop_duplicates("project_id", keep="first")
)


# %% [markdown]
# ## 2. Characterisation

# %%
gtr_df.info()

# %%
gtr_funded_projects.info()

# %%
# gtr_funds_.project_id

# %%
# # What fraction of row data is missing in GTR projects
# (gtr_df.isnull().sum() / len(gtr_df)).sort_values(ascending=False).round(3)

# %%
# gtr_df.iloc[]

# %% [markdown]
# ## 3. Analysis

# %% [markdown]
# ## Research projects

# %%
search_term = "heat pump"

# %%
from datetime import datetime


# %%
def is_term_present(search_term, docs):
    """ """
    return [search_term in doc for doc in docs]


def search_in_projects(search_term):
    bool_mask = is_term_present(search_term, gtr_docs)
    return gtr_projects[bool_mask].copy()


def get_project_funding(project_df):
    fund_cols = ["project_id", "id", "rel", "category", "amount", "currencyCode"]
    df = project_df.merge(
        gtr_funded_projects[fund_cols], on="project_id", how="left", validate="1:1"
    )
    return df


def convert_date_to_year(str_date):
    """String date to integer year"""
    if type(str_date) is str:
        return int(str_date[0:4])
    else:
        return str_date


def get_search_term_funding(search_term):
    projects = search_in_projects(search_term)
    projects_with_funding = get_project_funding(projects)
    projects_with_funding["year"] = projects_with_funding.start.apply(
        lambda date: convert_date_to_year(date)
    )
    return projects_with_funding


def get_breakdown_by_year(search_term):
    projects_with_funding = get_search_term_funding(search_term)
    df = (
        projects_with_funding.groupby("year")
        .agg({"project_id": "count", "amount": "sum"})
        .rename(columns={"project_id": "no_of_projects"})
        .reset_index()
    )
    df = df[df.year >= 2006]
    # To do - add missing years as 0s
    return df


def show_projects_by_year(search_term):
    df = get_breakdown_by_year(search_term)
    alt.Chart(df).mark_line(point=True).encode(x="year:T", y="no_of_projects:Q")


def show_funding_amount_by_year(search_term):
    df = get_breakdown_by_year(search_term)
    alt.Chart(df).mark_line(point=True).encode(x="year:T", y="amount:Q")


# %%
search_term = "heat pump"

df = get_breakdown_by_year(search_term)

alt.Chart(df).mark_line(point=True).encode(x="year:O", y="no_of_projects:Q")

# %%
alt.Chart(df).mark_line(point=True).encode(x="year:O", y="amount:Q")

# %%
check_columns = [
    "title",
    "grantCategory",
    "leadFunder",
    "potentialImpact",
    "year",
    "amount",
]
(
    get_search_term_funding(search_term)[check_columns].sort_values(
        ["year", "amount"], ascending=False
    )
).head(20)


# %% [markdown]
# ### Project landscape

# %%
len(project_docs)

# %%
project_docs

# %% [markdown]
# ## Crunchbase

# %%
# Select columns to check
columns = ["name", "short_description", "long_description"]
# Preprocess project text
cb_ = cb_df[columns].fillna("")
text_lists = [cb_[col].str.lower().to_list() for col in columns]
# Create project documents
cb_docs = list(create_documents(text_lists))


# %%
# Check if the text contains the search term
contains_term = is_term_present(search_term, cb_docs)

# %%
len(cb_df[contains_term])

# %%
cb_df_with_term = cb_df[contains_term].copy()
cb_df_with_term = cb_df_with_term[-cb_df_with_term.founded_on.isnull()]
cb_df_with_term["year"] = cb_df_with_term.founded_on.apply(
    lambda x: extract_year_from_cb(x)
)

# %%
cb_df_with_term.name


# %%
def extract_year_from_cb(str_date):
    if type(str_date) is str:
        return int(str_date[0:4])
    else:
        return str_date


# %%
cb_dff = (
    cb_df_with_term.groupby("year")
    .agg({"id": "count", "total_funding": "sum"})
    .reset_index()
)

# %%
cb_dff.year = pd.datetime(cb_dff.year)

# %%
cb_dff = cb_dff[cb_dff.year > 2008]

# %%
alt.Chart(cb_dff).mark_line().encode(x="year:Q", y="id:Q")

# %%
alt.Chart(cb_dff).mark_line().encode(x="year:T", y="total_funding:Q")

# %%
