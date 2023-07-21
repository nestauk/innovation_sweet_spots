# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

# # Visualisation of company landscape
#
# - Generate UMAP embeddings of vectors for each company
# - For visualisation purposes, assign each company to a cluster (rule-based or closest cluster centroid)
# - Use circle size to indicate venture funding amount (log scale)
# - Use colours for company subthemes
# - Try streamlit to make an interactive dashboard

# ## Import libraries and data

# +
import innovation_sweet_spots.utils.google_sheets as gs
from innovation_sweet_spots import PROJECT_DIR
import utils

from innovation_sweet_spots.getters.preprocessed import (
    get_preprocessed_crunchbase_descriptions,
)

import umap
import pandas as pd
from innovation_sweet_spots.utils.embeddings_utils import Vectors
import altair as alt
from innovation_sweet_spots.utils import plotting_utils as pu

# -

from innovation_sweet_spots.analysis.wrangling_utils import CrunchbaseWrangler

CB = CrunchbaseWrangler()

# # Download data with partially corrected subthemes
# subthemes_df = gs.download_google_sheet(
#     google_sheet_id=utils.AFS_GOOGLE_SHEET_ID_APRIL,
#     wks_name="list_v3",
# )


subthemes_df = pd.read_csv(
    PROJECT_DIR / "outputs/2023_childcare/finals/company_to_subtheme_v2023_05_16.csv"
).drop_duplicates(["cb_id", "subtheme_tag"])

subthemes_df.head(2)

len(subthemes_df)

# Download data with partially corrected subthemes
taxonomy_df = gs.download_google_sheet(
    google_sheet_id=utils.AFS_GOOGLE_SHEET_ID_APRIL,
    wks_name="taxonomy_final",
)

taxonomy_df

# Load a table with processed company descriptions
processed_texts = get_preprocessed_crunchbase_descriptions()

processed_texts.head(2)

print(processed_texts.sample().name.iloc[0])
print(processed_texts.sample().description.iloc[0])


# +
# Companies in the countries in scope
companies_in_countries = subthemes_df.query(
    "country in @utils.list_of_select_countries"
).cb_id.to_list()

# Final list of included companies
childcare_df = (
    subthemes_df.copy()
    .assign(
        keep=lambda df: df.keep.str.lower().apply(
            lambda x: True if x == "true" else False
        )
    )
    .query("cb_id in @companies_in_countries")
    .query("keep == True")
    .merge(processed_texts, left_on="cb_id", right_on="id", how="left")
    .drop(columns=["id", "name"])
)
# -

# ## Generate embeddings

descriptions_df = (
    processed_texts.query("id in @childcare_df.cb_id").drop_duplicates(subset=["id"])
).reset_index()


industries_df = CB.get_company_industries(
    descriptions_df[["id", "name"]], return_lists=True
)

# +
row = descriptions_df.sample().iloc[0]
text = row.description


def fix_industries(text, industries_df, cb_id):
    text_sentences = text.split(".")
    if "Industries" in text_sentences[-1]:
        industries = industries_df[industries_df.id == cb_id].industry.iloc[0]
        try:
            text_sentences[-1] = " Industries: {}".format(", ".join(industries))
        except:
            return text
        return ".".join(text_sentences)
    else:
        return text


fix_industries(text, industries_df, row.id)

# -

for i, row in descriptions_df.iterrows():
    descriptions_df.loc[i, "description"] = fix_industries(
        row.description, industries_df, row.id
    )

print(descriptions_df.sample().name.iloc[0])
print(descriptions_df.sample().description.iloc[0])

# +
# Define constants
EMBEDDINGS_DIR = PROJECT_DIR / "outputs/preprocessed/embeddings"
FILENAME = "childcare_startups_v2"

# Instansiate Vectors class
childcare_vectors = Vectors(
    model_name="all-mpnet-base-v2",
    vector_ids=None,
    filename=FILENAME,
    folder=EMBEDDINGS_DIR,
)
# -

# Make vectors
childcare_vectors.generate_new_vectors(
    new_document_ids=descriptions_df.id.values,
    texts=descriptions_df.description.values,
)

# +
# # Save vectors
# childcare_vectors.save_vectors()
# -

# ## Dedup multiple categories
#
# - Companies with tech + some other category
# - Companies with multiple other categories

# +
# All companies with only one assignment
single_cats_df = childcare_df[-childcare_df.cb_id.duplicated(keep=False)]

# Table with multiple categories
multiple_cats_df = childcare_df[childcare_df.cb_id.duplicated(keep=False)].sort_values(
    "cb_id"
)

# # Case: Remove rows that have "Tech" as one of the duplicated subthemes
# tech_ids = multiple_cats_df.query("subtheme_full == 'Tech'").cb_id.to_list()
# single_cats_nontech_df = multiple_cats_df.query("cb_id in @tech_ids and subtheme_full != 'Tech'")

# # Companies that still have duplicated subthemes
# multiple_cats_df = multiple_cats_df[-multiple_cats_df.cb_id.isin(single_cats_nontech_df.cb_id.to_list())]

# Case: Companies that have Content: General and a more specific Content subtheme
content_df = multiple_cats_df.query("theme == 'Content'")
content_df = content_df[content_df.duplicated(["cb_id"], keep=False)].sort_values(
    "cb_id"
)

content_ids = content_df.query("subtheme_full == 'Content: General'").cb_id.to_list()
single_cats_nongeneral_df = content_df.query(
    "cb_id in @content_ids and subtheme_full != 'Content: General'"
)
# -

single_cats_nongeneral_df

chidcare_dedup_df = pd.concat(
    [
        single_cats_df,
        # single_cats_nontech_df,
        single_cats_nongeneral_df,
    ],
    ignore_index=True,
)


chidcare_dedup_df.subtheme_full.value_counts()

# Drop if sbutheme_full is Other
chidcare_dedup_df = chidcare_dedup_df.query("subtheme_full != 'Other'")

# ## Visualise vectors

colour_map = dict(zip(taxonomy_df.subtheme_full, taxonomy_df.colour))

# +
umap_embeddings = umap.UMAP(
    n_neighbors=50,
    n_components=2,
    metric="euclidean",
    # random_state=21,
    random_state=1000,
).fit_transform(childcare_vectors.vectors)

umap_embeddings_df = pd.DataFrame(umap_embeddings, columns=["umap_1", "umap_2"]).assign(
    id=childcare_vectors.vector_ids
)

childcare_embeddings_df = (
    chidcare_dedup_df.merge(
        umap_embeddings_df, left_on="cb_id", right_on="id", how="left"
    )
    .assign(colour=lambda df: df.subtheme_full.map(colour_map))
    .drop("description", axis=1)
    .merge(
        descriptions_df[["id", "description"]],
        left_on="cb_id",
        right_on="id",
        how="left",
    )
)


# +
from matplotlib import pyplot as plt

# plot umap plot using plt
plt.figure(figsize=(5, 5))
plt.scatter(
    childcare_embeddings_df.umap_1,
    childcare_embeddings_df.umap_2,
    c=childcare_embeddings_df.colour,
    # cmap="Spectral",
    s=50,
    alpha=0.5,
)
# -

# Plot 2-d UMAP
fig = (
    alt.Chart(childcare_embeddings_df, width=800, height=600)
    .mark_circle(size=60)
    .encode(
        x="umap_1",
        y="umap_2",
        # define custom colour map
        color=alt.Color(
            "subtheme_full",
            legend=alt.Legend(title="Theme"),
            scale=alt.Scale(
                domain=list(colour_map.keys()), range=list(colour_map.values())
            ),
        ),
        tooltip=[
            "company_name",
            "homepage_url",
            "subtheme_full",
            "description",
            "total_funding_gbp",
        ],
        # add minimal size
        size=alt.Size("total_funding_gbp:Q", scale=alt.Scale(range=[50, 1000])),
    )
).interactive()

# save html
fig.save(
    str(PROJECT_DIR / "outputs/2023_childcare/figures/childcare_umap_v5.html"),
)

(
    childcare_embeddings_df[
        [
            "company_name",
            "country",
            "cb_url",
            "umap_1",
            "umap_2",
            "total_funding_gbp",
            "subtheme_full",
            "theme",
        ]
    ].to_csv(
        PROJECT_DIR / "outputs/2023_childcare/finals/startups_umap_v5.csv", index=False
    )
)

# ## Checking industries

#
