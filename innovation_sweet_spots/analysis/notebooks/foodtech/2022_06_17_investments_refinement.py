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
import numpy as np

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
import umap
import numpy as np

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
df = query.find_most_similar("restaurant tech").merge(labels).iloc[i : i + 20]
df

# %%
df.Category.to_list()

# %%
category_counts = (
    DR.company_labels.groupby(["Category", "label_type"], as_index=False)
    .agg(counts=("id", "count"))
    .sort_values("counts", ascending=False)
)


# %%
category_counts[category_counts.counts > 50]

# %%
# from itables import init_notebook_mode
# from itables import show

# init_notebook_mode(all_interactive=False)

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

# %% [markdown]
# ## User defined taxonomy

# %%
# Major category > Minor category > [dealroom_label, label_type]
# The minor category that has the same name as major category is a 'general' category
taxonomy = {
    "health and food": {
        "health (general)": [
            "health",
            "personal health",
            "health care",
            "wellness",
            "healthcare",
            "health and wellness",
        ],
        "diet": [
            "diet",
            "dietary supplements",
            "weight management",
        ],
        "nutrition": [
            "nutrition",
            "nutrition solution",
            "superfood",
            "healthy nutrition",
            "sports nutrition",
            "probiotics",
        ],
        "health issues": [
            "obesity",
            "diabetes",
            "disease",
            "allergies",
            "chronic disease",
            "gastroenterology",
        ],
        "health issues (other)": [
            "oncology",
            "immune system",
            "neurology",
            "mental health",
        ],
        "medicine and pharma": [
            "medical",
            "pharmaceutical",
            "therapeutics",
            "patient care",
            "drug development",
        ],
        "health tech": [
            "health platform",
            "medical devices",
            "medical device",
            "tech for patients",
            "digital healthcare",
            "health diagnostics",
            "health information",
            "medical technology",
            "healthtech",
            "digital health",
            "digital therapeutics",
        ],
    },
    "innovative food": {
        "alt protein": [
            "enabler of alternative proteins",
            "alternative protein",
            "meat substitute",
            "dairy substitute",
        ],
        "taste": [
            "taste",
            "flavor",
        ],
        "fermentation": ["fermentation"],
        "vegan": ["vegan"],
        "plant-based": ["plant-based"],
    },
    "logistics": {
        "logistics (general)": [
            "food logistics and delivery",
            "logistics and delivery",
            "logistic",
            "logistics",
            "logistics tech",
            "logistics solutions",
            "freight",
            "warehousing",
            "fleet management",
            "order management",
        ],
        "supply chain": [
            "supply chain management",
        ],
        "delivery": [
            "delivery",
            "food delivery platform",
            "food delivery service",
            "last-mile delivery",
            "shipping",
        ],
        "packaging": [
            "packaging and containers",
            "packaging",
            "sustainable packaging",
            "ecological packaging",
            "packaging solutions",
            "food packaging",
        ],
        "storage": [
            "storage",
        ],
        "meal kits": [
            "subscription boxes",
            "meal kits",
        ],
    },
    "cooking and catering": {
        "kitchen and cooking (general)": [
            "kitchen and cooking tech",
        ],
        "kitchen": [
            "kitchen",
            "dark kitchen",
        ],
        "restaurants and catering": [
            "catering",
            "restaurant tech",
            "restaurants management",
            "restaurant reservation",
        ],
        "cooking": [
            "cooking tech",
            "cooking",
            "chef",
            "recipes" "cook recipes",
        ],
    },
}


# %%
def create_taxonomy_dataframe(
    taxonomy: pd.DataFrame, DR: wu.DealroomWrangler = None
) -> pd.DataFrame:
    """
    Create a taxonomy dataframe from a dictionary
    """
    taxonomy_df = []
    for major in taxonomy.keys():
        for minor in taxonomy[major].keys():
            for label in taxonomy[major][minor]:
                taxonomy_df.append([major, minor, label])
    taxonomy_df = pd.DataFrame(taxonomy_df, columns=["Major", "Minor", "Category"])

    if DR is not None:
        # Number of companies for each label (NB: also accounts for multiple labels of different types with the same name)
        category_counts = (
            DR.company_labels.groupby(["Category", "label_type"], as_index=False)
            .agg(counts=("id", "count"))
            .sort_values("counts", ascending=False)
        )
        taxonomy_df = taxonomy_df.merge(
            category_counts[["Category", "label_type", "counts"]]
        )
    return taxonomy_df


# %%
taxonomy_df = create_taxonomy_dataframe(taxonomy, DR)

# %%
taxonomy_df

# %% [markdown]
# ## Helper functions

# %%
from collections import defaultdict
import itertools


# %%
def get_category_ids(taxonomy_df, DR, column="Category"):
    category_ids = defaultdict(set)
    for category in taxonomy_df[column].unique():
        ids = [
            DR.get_ids_by_labels(row.Category, row.label_type)
            for i, row in taxonomy_df.query(f"`{column}` == @category").iterrows()
        ]
        ids = set(itertools.chain(*ids))
        category_ids[category] = ids
    return category_ids


# %%
def get_category_ts(category_ids, DR):
    ind_ts = []
    for category in category_ids:
        ids = category_ids[category]
        ind_ts.append(
            au.cb_get_all_timeseries(
                DR.company_data.query("id in @ids"),
                (
                    DR.funding_rounds.query("id in @ids").query(
                        "`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES"
                    )
                ),
                period="year",
                min_year=2010,
                max_year=2022,
            )
            .assign(year=lambda df: df.time_period.dt.year)
            .assign(Category=category)
        )
    return pd.concat(ind_ts, ignore_index=True)


# %%
def get_company_counts(category_ids: dict):
    return pd.DataFrame(
        [(key, len(np.unique(list(category_ids[key])))) for key in category_ids],
        columns=["Category", "Number of companies"],
    )


# %%
def get_deal_counts(category_ids: dict):
    category_deal_counts = []
    for key in category_ids:
        ids = category_ids[key]
        deals = DR.funding_rounds.query("id in @ids").query(
            "`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES"
        )
        category_deal_counts.append((key, len(deals)))
    return pd.DataFrame(category_deal_counts, columns=["Category", "Number of deals"])


# %%
def get_trends(taxonomy_df, taxonomy_level, DR):
    category_ids = get_category_ids(taxonomy_df, DR, taxonomy_level)
    company_counts = get_company_counts(category_ids)
    category_ts = get_category_ts(category_ids, DR)

    values_title_ = "raised_amount_gbp_total"
    values_title = "Growth"
    category_title = "Category"
    colour_title = category_title
    horizontal_title = "year"

    if taxonomy_level == "Category":
        tax_levels = ["Category", "Minor", "Major"]
    if taxonomy_level == "Minor":
        tax_levels = ["Minor", "Major"]
    if taxonomy_level == "Major":
        tax_levels = ["Major"]

    return (
        utils.get_magnitude_vs_growth(
            category_ts,
            value_column=values_title_,
            time_column=horizontal_title,
            category_column=category_title,
        )
        .assign(growth=lambda df: df.Growth / 100)
        .merge(get_deal_counts(category_ids), on="Category")
        .merge(company_counts, on="Category")
        .merge(
            taxonomy_df[tax_levels].drop_duplicates(taxonomy_level),
            how="left",
            left_on="Category",
            right_on=taxonomy_level,
        )
    )


# %%
def fig_growth_vs_magnitude(
    magnitude_vs_growth,
    colour_field,
    text_field,
    legend=alt.Legend(),
):
    title_text = "Foodtech trends (2017-2021)"
    subtitle_text = [
        "Data: Dealroom. Showing data on early stage deals (eg, series funding)",
        "Late stage deals, such as IPOs, acquisitions, and debt not included.",
    ]

    fig = (
        alt.Chart(
            magnitude_vs_growth,
            width=400,
            height=400,
        )
        .mark_circle(size=50)
        .encode(
            x=alt.X(
                "Magnitude:Q",
                axis=alt.Axis(title=f"Average yearly raised amount (million GBP)"),
                # scale=alt.Scale(type="linear"),
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y(
                "growth:Q",
                axis=alt.Axis(title="Growth", format="%"),
                # axis=alt.Axis(
                #     title=f"Growth between {start_year} and {end_year} measured by number of reviews"
                # ),
                # scale=alt.Scale(domain=(-.100, .300)),
                #             scale=alt.Scale(type="log", domain=(.01, 12)),
            ),
            #         size="Number of companies:Q",
            color=alt.Color(f"{colour_field}:N", legend=legend),
            tooltip=[
                "Category",
                alt.Tooltip(
                    "Magnitude", title=f"Average yearly raised amount (million GBP)"
                ),
                alt.Tooltip("growth", title="Growth", format=".0%"),
            ],
        )
        .properties(
            title={
                "anchor": "start",
                "text": title_text,
                "subtitle": subtitle_text,
                "subtitleFont": pu.FONT,
            },
        )
    )

    text = fig.mark_text(align="left", baseline="middle", font=pu.FONT, dx=7).encode(
        text=text_field
    )

    fig_final = (
        (fig + text)
        .configure_axis(
            grid=False,
            gridDash=[1, 7],
            gridColor="white",
            labelFontSize=pu.FONTSIZE_NORMAL,
            titleFontSize=pu.FONTSIZE_NORMAL,
        )
        .configure_legend(
            titleFontSize=pu.FONTSIZE_NORMAL,
            labelFontSize=pu.FONTSIZE_NORMAL,
        )
        .configure_view(strokeWidth=0)
    )

    return fig_final


# %%
def fig_category_growth(
    magnitude_vs_growth_filtered,
    colour_field,
):
    """ """
    fig = (
        alt.Chart(
            (
                magnitude_vs_growth_filtered.assign(
                    Increase=lambda df: df.growth > 0
                ).assign(Magnitude_log=lambda df: np.log10(df.Magnitude))
            ),
            width=300,
            height=450,
        )
        .mark_circle(color=pu.NESTA_COLOURS[0], opacity=1)
        .encode(
            x=alt.X(
                "growth:Q",
                axis=alt.Axis(
                    format="%",
                    title="Growth",
                    labelAlign="center",
                    labelExpr="datum.value < -1 ? null : datum.label",
                ),
                #             scale=alt.Scale(domain=(-1, 37)),
            ),
            y=alt.Y(
                "Category:N",
                sort="-x",
                axis=alt.Axis(title="Category"),
            ),
            size=alt.Size(
                "Magnitude",
                title="Yearly investment (million GBP)",
                legend=alt.Legend(orient="top"),
                scale=alt.Scale(domain=[100, 4000]),
            ),
            color=alt.Color(
                colour_field,
            ),
            # size="cluster_size:Q",
            #         color=alt.Color(f"{colour_title}:N", legend=None),
            tooltip=[
                alt.Tooltip("Category:N", title="Category"),
                alt.Tooltip(
                    "Magnitude:Q",
                    format=",.3f",
                    title="Average yearly investment (million GBP)",
                ),
                "Number of companies",
                "Number of deals",
                alt.Tooltip("growth:Q", format=",.0%", title="Growth"),
            ],
        )
    )

    # text = fig.mark_text(align="left", baseline="middle", font=pu.FONT, dx=7).encode(
    #     text='text_label:N'
    # )

    # fig_final = (
    #     (fig)
    #     .configure_axis(
    #         gridDash=[1, 7],
    #         gridColor="grey",
    #         labelFontSize=pu.FONTSIZE_NORMAL,
    #         titleFontSize=pu.FONTSIZE_NORMAL,
    #     )
    #     .configure_legend(
    #         labelFontSize=pu.FONTSIZE_NORMAL - 1,
    #         titleFontSize=pu.FONTSIZE_NORMAL - 1,
    #     )
    #     .configure_view(strokeWidth=0)
    #     #     .interactive()
    # )

    return pu.configure_titles(pu.configure_axes(fig), "", "")


# %%
def fig_size_vs_magnitude(
    magnitude_vs_growth_filtered,
    colour_field,
):
    fig = (
        alt.Chart(
            magnitude_vs_growth_filtered,
            width=500,
            height=450,
        )
        .mark_circle(color=pu.NESTA_COLOURS[0], opacity=1, size=50)
        .encode(
            x=alt.X(
                "Number of companies:Q",
            ),
            y=alt.Y(
                "Magnitude:Q",
                axis=alt.Axis(title=f"Average yearly raised amount (million GBP)"),
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color(colour_field),
            # size="cluster_size:Q",
            #         color=alt.Color(f"{colour_title}:N", legend=None),
            tooltip=[
                alt.Tooltip("Category:N", title="Category"),
                alt.Tooltip(
                    "Magnitude:Q",
                    format=",.3f",
                    title="Average yearly investment (million GBP)",
                ),
                alt.Tooltip("growth:Q", format=",.0%", title="Growth"),
                "Number of companies",
                "Number of deals",
            ],
        )
    )

    return pu.configure_titles(pu.configure_axes(fig), "", "")


# %% [markdown]
# ## Minor categories (medium granularity)

# %%
# Initialise a Dealroom wrangler instance
importlib.reload(wu)
DR = wu.DealroomWrangler()

# %%
magnitude_vs_growth = get_trends(taxonomy_df, "Minor", DR)
magnitude_vs_growth

# %%
fig_growth_vs_magnitude(
    magnitude_vs_growth, colour_field="Major", text_field="Minor"
).interactive()

# %%
fig_category_growth(magnitude_vs_growth, colour_field="Major")

# %%
fig_size_vs_magnitude(magnitude_vs_growth, colour_field="Major")

# %%
# category='meal kits'
# pu.cb_investments_barplot(
#     category_ts.query("Category == @category"),
#     y_column="raised_amount_gbp_total",
#     y_label="Raised amount (million GBP)",
#     x_label="Year",
# )

# %% [markdown]
# ## Tags (most granular categories)

# %%
magnitude_vs_growth = get_trends(taxonomy_df, "Category", DR)

# %%
magnitude_vs_growth_filtered = magnitude_vs_growth.query(
    "`Number of companies` > 10"
).query("`Number of deals` > 10")

# %%
fig_growth_vs_magnitude(
    magnitude_vs_growth_filtered,
    colour_field="Minor",
    text_field="Category",
).interactive()


# %%

# %%

# %%
category = "meal kits"
pu.cb_investments_barplot(
    category_ts.query("Category == @category"),
    y_column="raised_amount_gbp_total",
    y_label="Raised amount (million GBP)",
    x_label="Year",
)

# %%

# %% [markdown]
# # Segmenting the categories

# %%
clusters = cluster_analysis_utils.hdbscan_clustering(v_labels.vectors)

# %%
df_clusters = (
    pd.DataFrame(clusters, columns=["cluster", "probability"])
    .astype({"cluster": int, "probability": float})
    .assign(text=v_labels.vector_ids)
    .merge(labels, how="left")
    .merge(category_counts, how="left")
    .drop_duplicates("text")
)

# %%
df_clusters

# %%
extra_stopwords = ["tech", "technology", "food"]
stopwords = cluster_analysis_utils.DEFAULT_STOPWORDS + extra_stopwords


cluster_keywords = cluster_analysis_utils.cluster_keywords(
    df_clusters.text.apply(
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
len(v_labels.vector_ids)

# %%
df_viz = (
    pd.DataFrame(v_labels.vector_ids, columns=["text"])
    .merge(df_clusters, how="left")
    .assign(
        x=embedding[:, 0],
        y=embedding[:, 1],
        #         cluster=df_clusters.cluster,
        #         cluster_prob=df_clusters.probability,
        cluster_name=df_clusters.cluster.apply(lambda x: str(cluster_keywords[x])),
        cluster_str=lambda df: df.cluster.astype(str),
        log_counts=lambda df: np.log10(df.counts),
    )
    .sort_values(["cluster", "counts"], ascending=False)
)

# %%
df_viz

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
        color="cluster_str",
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
from innovation_sweet_spots import PROJECT_DIR

df_viz.to_csv(PROJECT_DIR / "outputs/foodtech/interim/dealroom_labels.csv", index=False)

# %%
# c=34
# df_clusters.query("cluster == @c").label

# %%
ids = DR.company_labels.query("Category == 'taste'").id.to_list()
DR.company_data.query("id in @ids")

# %%
ids = DR.company_labels.query("Category == 'meal kits'").id.to_list()
DR.company_labels.query("id in @ids").groupby("Category").agg(
    counts=("id", "count")
).sort_values("counts", ascending=False)

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
# # Visualise using altair
# fig = (
#     alt.Chart(df_viz, width=1000, height=1000)
#     .mark_circle(size=20, color=pu.NESTA_COLOURS[1])
#     .encode(
#         x=alt.X("x", axis=None),
#         y=alt.Y("y", axis=None),
#         tooltip=list(df_viz.columns),
#         #         color="Primary Category:N",
#         color="in_category:N",
#         href="WEBSITE",
#         #         size="Hours per Week",
#     )
# )

# # text = (
# #     alt.Chart(centroids)
# #     .mark_text(font=pu.FONT)
# #     .encode(x=alt.X("x_c:Q"), y=alt.Y("y_c:Q"), text=alt.Text("keywords"))
# # )

# fig_final = (
#     (fig)
#     .configure_axis(
#         # gridDash=[1, 7],
#         gridColor="white",
#     )
#     .configure_view(strokeWidth=0, strokeOpacity=0)
#     .properties(
#         title={
#             "anchor": "start",
#             "text": ["Landscape of companies"],
#             "subtitle": "",
#             #             [
#             #                 "Each circle is a course; courses with similar titles will be closer on this map",
#             #                 "Press Shift and click on a circle to go the course webpage",
#             #             ],
#             "subtitleFont": pu.FONT,
#         },
#     )
#     .interactive()
# )

fig_final

# %%
