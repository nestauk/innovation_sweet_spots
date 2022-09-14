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
# # Analysing hansard debates

# %%
from innovation_sweet_spots.getters import hansard
import importlib

importlib.reload(hansard)

# %%
from innovation_sweet_spots.analysis.query_terms import QueryTerms
from innovation_sweet_spots.getters.preprocessed import get_hansard_corpus

# %%
import innovation_sweet_spots.analysis.query_terms as query_terms

importlib.reload(query_terms)

# %%
from innovation_sweet_spots.utils.io import load_pickle
from innovation_sweet_spots import PROJECT_DIR

tech_area_terms = load_pickle(
    PROJECT_DIR / "outputs/foodtech/interim/foodtech_search_terms.pickle"
)
# tech_area_terms['Biomedical'].append(['obese'])

# %% [markdown]
# # Speeches data

# %%
df_debates = hansard.get_debates()

# %%
df_debates = df_debates.drop_duplicates("id", keep="first")

# %%
assert len(df_debates.id.unique()) == len(df_debates)

# %%
# df_debates[df_debates.id.duplicated(keep=False)].sort_values('id')

# %%
len(df_debates)

# %% [markdown]
# # Keyword search

# %%
tech_areas_to_check = list(tech_area_terms.keys())[:-3]

# %%
hansard_corpus = get_hansard_corpus()

# %%
Query_hansard = QueryTerms(corpus=hansard_corpus)

# %%
food_hits = Query_hansard.find_matches(
    tech_area_terms["Food terms"], return_only_matches=True
)

# %%
gtr_query_results, gtr_all_hits = query_terms.get_document_hits(
    Query_hansard, tech_area_terms, tech_areas_to_check, food_hits
)

# %% [markdown]
# 2022-09-14 09:54:24,455 - root - INFO - Found 31 documents with search terms ['food', 'reformulation']
#
# 2022-09-14 09:54:26,051 - root - INFO - Found 53 documents with search terms ['food', 'reformulat']
#
#
# 2022-09-14 09:55:34,497 - root - INFO - Found 1471 documents with search terms ['obesity']
#
# 2022-09-14 09:55:35,512 - root - INFO - Found 225 documents with search terms ['overweight']
#
# 2022-09-14 09:55:37,061 - root - INFO - Found 301 documents with search terms ['obese']
#
# 2022-09-14 09:56:04,336 - root - INFO - Found 108 documents with search terms ['food environment']
#

# %%
hansard_query_export = gtr_query_results.merge(
    df_debates[["id", "speech", "speakername", "year"]], on="id"
)

# %%
# for p in query_df_.speech:
#     print(p)
#     print('---')

# %%
hansard_query_export.to_csv(
    PROJECT_DIR
    / "outputs/foodtech/interim/public_discourse/hansard_hits_v2022_09_14.csv",
    index=False,
)

# %%
