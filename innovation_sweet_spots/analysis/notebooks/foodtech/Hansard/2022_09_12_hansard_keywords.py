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

# %%
tech_area_terms["Supply chain"] = tech_area_terms["Supply chain"][1:]
tech_area_terms["Kitchen tech"] = tech_area_terms["Kitchen tech"][1:]

# %%
from innovation_sweet_spots.getters.google_sheets import get_foodtech_search_terms

df_search_terms = get_foodtech_search_terms()

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

# %% [markdown]
# ## Baseline speeches

# %%
import altair as alt

# %%
hansard_baseline = df_debates.groupby("year", as_index=False).agg(
    total_counts=("id", "count")
)

# %%
alt.Chart(hansard_baseline).mark_line().encode(x="year", y="total_counts")

# %% [markdown]
# ## Trends
# - Time series across years of speeches per main category
# - Time series across years of speeches per consolidated categories

# %%
# Add taxonomy to the hits
hansard_query_export_ = hansard_query_export.merge(
    df_search_terms[["Tech area", "Terms", "Sub Category", "Category"]].drop_duplicates(
        "Tech area"
    ),
    how="left",
    left_on="tech_area",
    right_on="Tech area",
)

# %%
hansard_query_export_.groupby(["Category", "Sub Category"]).agg(
    counts=("id", "count")
).reset_index()

# %%
hansard_query_export_.groupby(["Category", "Sub Category", "Tech area"]).agg(
    counts=("id", "count")
).reset_index()

# %%
# Export data
ts_category = (
    hansard_query_export_.groupby(["year", "Category"])
    .agg(counts=("id", "count"))
    .reset_index()
    .merge(hansard_baseline, how="left", on="year")
    .assign(fraction=lambda df: df.counts / df.total_counts)
)

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_category)
    .mark_line()
    .encode(
        x=alt.X("year:O"),
        y=alt.Y("fraction:Q", sort="-x", scale=alt.Scale(type=scale)),
        # size=alt.Size('magnitude'),
        color="Category",
        tooltip=["year", "counts", "Category"],
    )
)
fig

# %%
# for i, row in hansard_query_export_.query('Category=="Innovative food"').iterrows():
#     print(row.year, row.speech)
#     print('/n')

# %%
# build a time series
# impute empty years

category_ts = []
for tech_area in categories_to_check:
    df = research_project_funding.query("Category == @tech_area")
    df_ts = au.gtr_get_all_timeseries_period(
        df, period="year", min_year=2010, max_year=2022, start_date_column="start_date"
    ).assign(tech_area=tech_area)
    tech_area_ts.append(df_ts)
tech_area_ts = pd.concat(tech_area_ts, ignore_index=False)

# %% [markdown]
# ## Other checks

# %%
import re
import ast


def extract_mention_sentence(text, search_term, any_term=False):
    text = re.sub("hon\.", "hon", text)
    text = re.sub("Hon\.", "Hon", text)
    sents = text.lower().split(".")
    sents_with_terms = []
    for sent in sents:
        if any_term:
            has_mention = False
            for s in search_term:
                has_mention = has_mention or (s in sent)
        else:
            has_mention = True
            for s in search_term:
                has_mention = has_mention and (s in sent)
        if has_mention == True:
            sents_with_terms.append(sent)
    return sents_with_terms


# %%
# search_term = ['takeaway service']
# search_term = ['takeaway']
# search_term = ['supply chain', 'food', 'innovation']
# search_term = ['meal kit']
# search_term = ['obesity']
# search_term = ['food waste']
# search_term = ['reformulation', 'food']
search_term = ["food environment"]
hits = Query_hansard.find_matches([search_term], return_only_matches=True)
hits = hits.merge(df_debates[["id", "speech", "speakername", "year"]], on="id")

# %%
hits.groupby("year").agg(counts=("id", "count"))

# %%
hits["sents"] = hits.speech.apply(
    lambda x: extract_mention_sentence(x, search_term, any_term=True)
)

# %%
for i, row in hits.iterrows():
    if int(row.year) > 2000:
        print(row.year, row.speech)
        print("")

# %% [markdown]
# ## Checking categories

# %%
sents = []
for i, row in hansard_query_export_.iterrows():
    sents.append(
        [
            extract_mention_sentence(row.speech, s, any_term=False)
            for s in ast.literal_eval(row.found_terms)
        ]
    )

# %%
hansard_query_export_["sents"] = sents

# %%
cat = "Innovative food"
for i, row in (
    hansard_query_export_.sort_values("year").query("Category == @cat").iterrows()
):
    if int(row.year) > 2000:
        print(row.year, row.sents)
        print("")

# %%
