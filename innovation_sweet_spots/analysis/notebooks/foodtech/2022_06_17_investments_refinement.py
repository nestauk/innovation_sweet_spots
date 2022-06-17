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
import innovation_sweet_spots.analysis.analysis_utils as au
from innovation_sweet_spots.utils import plotting_utils as pu
import utils
import innovation_sweet_spots.utils.text_cleaning_utils as tcu

importlib.reload(wu)
import altair as alt
import pandas as pd

COLUMN_CATEGORIES = wu.dealroom.COLUMN_CATEGORIES

# %%
# Functionality for saving charts
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver(path=alt_save.FIGURE_PATH + "/foodtech")

# %%
# Initialise a Dealroom wrangler instance
importlib.reload(wu)
DR = wu.DealroomWrangler()

# Number of companies
len(DR.company_data)

# %% [markdown]
# ## Explore label embeddings

# %%
import innovation_sweet_spots.getters.dealroom as dlr
from innovation_sweet_spots.utils import cluster_analysis_utils
import innovation_sweet_spots.utils.embeddings_utils as eu

# %%
v_labels = dlr.get_label_embeddings()

# %%
query = eu.QueryEmbeddings(
    vectors=v_labels.vectors, texts=v_labels.vector_ids, model=v_labels.model
)

# %%
labels = DR.labels.assign(text=lambda df: df.Category.apply(tcu.clean_dealroom_labels))

# %%
pd.set_option("max_colwidth", 200)
ids = DR.company_labels.query("Category=='compounding'").id.to_list()
DR.company_data.query("id in @ids")[["NAME", "TAGLINE", "WEBSITE"]]

# %%
i = 0
df = query.find_most_similar("nutrition").merge(labels).iloc[i : i + 20]
df

# %%
df.Category.to_list()

# %%
health_Weight = [
    "obesity",
    "metabolism",
    "diet",
    "nutrition",
    "weight management",
    "healthy nutrition",
    "healthy shake",
    "dietary guidance",
    "diet personalization",
    "nutrition tracking",
    "nutrition solution",
    "macronutrients",
    "micronutrients",
    "health supplements",
    "medical food",
]

health_Comorbid = [
    "diabetes",
    "chronic disease",
    "health issues",
]

health_Sports = [
    "fitness",
]

# %%
clusters = cluster_analysis_utils.hdbscan_clustering(v_labels.vectors)

# %%
df_clusters = (
    pd.DataFrame(clusters, columns=["cluster", "probability"])
    .astype({"cluster": int, "probability": float})
    .assign(label=v_labels.vector_ids)
)

# %%
df_clusters.head(2)

# %%
extra_stopwords = ["tech", "technology", "food"]
stopwords = cluster_analysis_utils.DEFAULT_STOPWORDS + extra_stopwords


cluster_keywords = cluster_analysis_utils.cluster_keywords(
    df_clusters.label.apply(
        lambda x: cluster_analysis_utils.simple_preprocessing(x, stopwords)
    ).to_list(),
    df_clusters.cluster.to_list(),
    11,
    max_df=0.7,
    min_df=0.1,
)

# %%
umap_viz_params = {
    "n_components": 2,
    "n_neighbors": 10,
    "min_dist": 0.5,
    "spread": 0.5,
}

reducer_2d = umap.UMAP(random_state=1, **umap_viz_params)
embedding = reducer_2d.fit_transform(v_labels.vectors)

# %%
embedding.shape

# %%
df_viz = (
    pd.DataFrame(v_labels.vector_ids, columns=["Category"])
    .merge(
        (
            DR.company_labels.assign(
                Category=lambda df: df.Category.apply(tcu.clean_dealroom_labels)
            )
            .groupby("Category", as_index=False)
            .agg(counts=("id", "count"), label_type=("label_type", lambda x: x.iloc[0]))
        ),
        how="left",
    )
    .assign(
        x=embedding[:, 0],
        y=embedding[:, 1],
        cluster=df_clusters.cluster,
        cluster_prob=df_clusters.probability,
        cluster_name=df_clusters.cluster.apply(lambda x: str(cluster_keywords[x])),
        log_counts=lambda df: np.log10(df.counts),
    )
)

# %%
# Visualise using altair
fig = (
    alt.Chart(df_viz, width=600, height=500)
    .mark_circle(size=20, color=pu.NESTA_COLOURS[0])
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        size="log_counts",
        tooltip=list(df_viz.columns),
        #         color="label_type",
        color="cluster_name",
    )
)

# text = (
#     alt.Chart(centroids)
#     .mark_text(font=pu.FONT)
#     .encode(x=alt.X("x_c:Q"), y=alt.Y("y_c:Q"), text=alt.Text("keywords"))
# )

fig_final = (
    #     (fig + text)
    (fig)
    .configure_axis(
        # gridDash=[1, 7],
        gridColor="white",
    )
    .configure_view(strokeWidth=0, strokeOpacity=0)
    .properties(
        title={
            "anchor": "start",
            "text": ["Company labels"],
            "subtitle": [
                "All industry, sub-industry and tag labels.",
                "1328-d sentence embeddings > 2-d UMAP",
            ],
            "subtitleFont": pu.FONT,
        },
    )
    .interactive()
)

fig_final

# %%
# c=34
# df_clusters.query("cluster == @c").label

# %% [markdown]
# ## Check health companies in detail

# %%
df_health = DR.get_companies_by_industry("health")

# %%
len(df_health)

# %%
df_health_labels = DR.company_tags.query("id in @df_health.id.to_list()")


# %%
# DR.company_labels.query("Category=='health'")

# %%
health_label_counts = (
    df_health_labels.groupby("TAGS")
    .agg(counts=("id", "count"))
    .sort_values("counts", ascending=False)
)

# %%
health_label_counts.head(15)

# %%
chosen_cat = DR.company_labels.query("Category in @health_Weight").id.to_list()

# %%
# company_labels_list = (
#     DR.company_labels
#     .assign(Category = lambda df: df.Category.apply(tcu.clean_dealroom_labels))
#     .groupby("id")["Category"].apply(list)
# )

# %%
importlib.reload(dlr)
v_vectors = dlr.get_company_embeddings(
    filename="foodtech_may2022_companies_tagline_labels"
)

# %%
# Reduce the embedding to 2 dimensions
reducer_low_dim = umap.UMAP(
    n_components=2,
    random_state=333,
    n_neighbors=10,
    min_dist=0.5,
    spread=0.5,
)
embedding = reducer_low_dim.fit_transform(v_vectors.vectors)

# %%
df_viz = (
    DR.company_data[["id", "NAME", "TAGLINE", "WEBSITE"]]
    .copy()
    .assign(x=embedding[:, 0])
    .assign(y=embedding[:, 1])
    .merge(
        pd.DataFrame(
            DR.company_labels.groupby("id")["Category"].apply(list)
        ).reset_index()
    )
    .assign(in_category=lambda df: df.id.isin(chosen_cat))
)

# %%
alt.data_transformers.disable_max_rows()

# %%
# Visualise using altair
fig = (
    alt.Chart(df_viz, width=1000, height=1000)
    .mark_circle(size=20, color=pu.NESTA_COLOURS[1])
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        tooltip=list(df_viz.columns),
        #         color="Primary Category:N",
        color="in_category:N",
        href="WEBSITE",
        #         size="Hours per Week",
    )
)

# text = (
#     alt.Chart(centroids)
#     .mark_text(font=pu.FONT)
#     .encode(x=alt.X("x_c:Q"), y=alt.Y("y_c:Q"), text=alt.Text("keywords"))
# )

fig_final = (
    (fig)
    .configure_axis(
        # gridDash=[1, 7],
        gridColor="white",
    )
    .configure_view(strokeWidth=0, strokeOpacity=0)
    .properties(
        title={
            "anchor": "start",
            "text": ["Landscape of companies"],
            "subtitle": "",
            #             [
            #                 "Each circle is a course; courses with similar titles will be closer on this map",
            #                 "Press Shift and click on a circle to go the course webpage",
            #             ],
            "subtitleFont": pu.FONT,
        },
    )
    .interactive()
)

fig_final

# %%
