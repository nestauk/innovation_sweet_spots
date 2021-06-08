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
from innovation_sweet_spots.getters.inputs import get_gtr_projects, get_cb_data
import pandas as pd
import time
import numpy as np

# %% [markdown]
# ## 1. Setup the analysis

# %%
search_term = "heat pump"

# %% [markdown]
# ### Load in the data

# %%
import importlib

# %%
from innovation_sweet_spots.getters import inputs

importlib.reload(inputs)

# %%
gtr = inputs.get_gtr_projects(
    fpath=inputs.GTR_PATH.parent / "gtr_test.json",
    table_wildcards=["gtr_funds"],
    fields=["id"],
    use_cached=False,
)

# %%
# %%capture
# gtr = get_gtr_projects()
cb_tables = get_cb_data()

# %%
gtr_df = pd.DataFrame(gtr)

# %%
gtr_df.info()

# %% [markdown]
# ## 2. Characterisation

# %%
gtr_df.loc[gtr_df.end == "0001-01-01T00:00:00", "end"] = np.nan
gtr_df.loc[gtr_df.start == "0001-01-01T00:00:00", "start"] = np.nan
gtr_df.loc[gtr_df.created == "0001-01-01T00:00:00", "created"] = np.nan

# %%
# What fraction of row data is missing in GTR projects
(gtr_df.isnull().sum() / len(gtr_df)).sort_values(ascending=False).round(3)


# %%
# gtr_df.iloc[]

# %% [markdown]
# ## 3. Analysis

# %%
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


# %% [markdown]
# ## Research projects

# %%
# Select columns to check
columns = ["title", "abstractText", "techAbstractText"]
# Preprocess project text
gtr_df = gtr_df.fillna("")
text_lists = [gtr_df[col].str.lower().to_list() for col in columns]
# Create project documents
project_docs = list(create_documents(text_lists))


# %%
@timeit
def is_term_present(search_term, docs):
    """ """
    return [search_term in doc for doc in docs]


# %%
# Check if the text contains the search term
contains_term = is_term_present(search_term, project_docs)

# %%
# Number of projects with this term
sum(contains_term)

# %%
gtr_df_with_term = gtr_df[contains_term]

# %%
gtr_df_with_term.iloc[0]

# %%
