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
# # Rapid exploration

# %%
import innovation_sweet_spots.analysis.wrangling_utils as wu
import importlib
import innovation_sweet_spots.analysis.analysis_utils as au
from innovation_sweet_spots.utils import plotting_utils as pu
import utils

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

# %%
# Number of companies
len(DR.company_data)

# %%
DR.company_data.head()

# %%
# Number of funding rounds
len(DR.funding_rounds)

# %%
DR.funding_rounds.head(2)

# %%
# Currencies that are not covered by our conversion approach
Converter = wu.CurrencyConverter()
COLUMN = "EACH ROUND CURRENCY"
all_dealroom_currencies = set(DR.explode_dealroom_table(COLUMN)[COLUMN].unique())
all_dealroom_currencies.remove("n/a")
curr = all_dealroom_currencies.difference(Converter.currencies)
curr

# %%
# Deals that we'll lose because of currency
DR.funding_rounds["EACH ROUND CURRENCY"].isin(curr).sum()

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

# %% [markdown]
# # Rapid plots
#
# NB, there might be differences between our data snapshot and online database. Small differences due to exchange rates, and larger differences due to the database having more up to date deal information (or sometimes for some reasoning missing some deals on their foodtech app).

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

INDUSTRIES = [
    "health",
]

# %%
ind_ts = []
for ind in SUBINDUSTRIES:
    org_df = DR.get_companies_by_subindustry(ind)
    deals_df = DR.get_rounds_by_subindustry(ind).query(
        "`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES"
    )
    ind_ts.append(
        au.cb_get_all_timeseries(
            org_df, deals_df, period="year", min_year=2010, max_year=2022
        )
        .assign(year=lambda df: df.time_period.dt.year)
        .assign(Category=ind)
    )

for ind in INDUSTRIES:
    ind_ts.append(
        au.cb_get_all_timeseries(
            DR.get_companies_by_industry(ind),
            (
                DR.get_rounds_by_industry(ind).query(
                    "`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES"
                )
            ),
            period="year",
            min_year=2010,
            max_year=2022,
        )
        .assign(year=lambda df: df.time_period.dt.year)
        .assign(Category=ind)
    )


ind_ts = pd.concat(ind_ts, ignore_index=True)

# %%
company_counts = pd.concat(
    [
        (
            DR.company_subindustries.groupby("SUB INDUSTRIES")
            .agg(counts=("id", "count"))
            .reset_index()
            .rename(columns={"SUB INDUSTRIES": "Category"})
        ),
        (
            DR.company_industries.groupby("INDUSTRIES")
            .agg(counts=("id", "count"))
            .reset_index()
            .rename(columns={"INDUSTRIES": "Category"})
        ),
    ],
    ignore_index=True,
)

# %%
health_ids = DR.get_companies_by_industry("health").id.to_list()
(
    DR.company_subindustries.query("id in @health_ids")
    .groupby("SUB INDUSTRIES")
    .agg(counts=("id", "count"))
    .sort_values("counts", ascending=False)
)

# %%
# DR.get_companies_by_industry('health')

# %%
ind_ts.head()

# %%
importlib.reload(au)
importlib.reload(utils)

# %%
# Longer term trend (2017 -> 2021)
values_title_ = "raised_amount_gbp_total"
values_title = "Growth"
category_title = "Category"
colour_title = category_title
horizontal_title = "year"

df = (
    utils.get_estimates(
        ind_ts,
        value_column=values_title_,
        time_column=horizontal_title,
        category_column=category_title,
        estimate_function=au.smoothed_growth,
        year_start=2017,
        year_end=2021,
    )
    .assign(growth=lambda df: (df[values_title_] / 100).round(2))
    .rename(columns={"growth": values_title})
)
df

# %%
importlib.reload(utils)
magnitude_vs_growth = (
    utils.get_magnitude_vs_growth(
        ind_ts,
        value_column=values_title_,
        time_column=horizontal_title,
        category_column=category_title,
    )
    .merge(company_counts, on="Category")
    .rename(columns={"counts": "Number of companies"})
)
magnitude_vs_growth

# %%
DR.company_subindustries

# %%
title_text = "Foodtech trends (2017-2021)"
subtitle_text = [
    "Data: Dealroom. Showing data on early stage deals (eg, series funding)",
    "Late stage deals, such as IPOs, acquisitions, and debt not included.",
]

fig = (
    alt.Chart(
        (magnitude_vs_growth.assign(Growth=lambda df: df.Growth / 100)),
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
            "Growth:Q",
            axis=alt.Axis(format="%"),
            # axis=alt.Axis(
            #     title=f"Growth between {start_year} and {end_year} measured by number of reviews"
            # ),
            # scale=alt.Scale(domain=(-.100, .300)),
            #             scale=alt.Scale(type="log", domain=(.01, 12)),
        ),
        #         size="Number of companies:Q",
        color=alt.Color(f"{colour_title}:N", legend=None),
        tooltip=["Category", "Magnitude", "Growth"],
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
    text=colour_title
)

fig_final = (
    (fig + text)
    .configure_axis(
        gridDash=[1, 7],
        gridColor="white",
    )
    .configure_view(strokeWidth=1)
)

fig_final

# %%
importlib.reload(utils)
table_name = "magnitude_growth"
# utils.save_data_table(reviews_per_year_by_user, table_name)
AltairSaver.save(fig_final, table_name, filetypes=["html", "png"])

# %%
values_title = "Number of companies"
labels_title = "Category"
tooltip = [labels_title, values_title]
color = pu.NESTA_COLOURS[0]

chart_title = "Number of companies in each category"
chart_subtitle = "Note: One company can be in multiple categories"

fig = (
    alt.Chart(
        magnitude_vs_growth,
        width=300,
        height=300,
    )
    .mark_bar(color=color)
    .encode(
        x=alt.X(
            f"{values_title}:Q", title=values_title, scale=alt.Scale(domain=(0, 4000))
        ),
        y=alt.Y(
            f"{labels_title}:N",
            title=labels_title,
            sort="-x",
            axis=alt.Axis(labelLimit=200),
        ),
        tooltip=tooltip,
    )
    .properties(
        title={
            "anchor": "start",
            "text": chart_title,
            "subtitle": chart_subtitle,
            "subtitleFont": pu.FONT,
        },
    )
    .configure_axis(
        gridDash=[1, 7],
        gridColor="grey",
    )
    .configure_view(strokeWidth=0)
)
fig

# %%
importlib.reload(utils)
table_name = "number_of_companies_by_category"
# utils.save_data_table(reviews_per_year_by_user, table_name)
AltairSaver.save(fig, table_name, filetypes=["html", "png"])

# %%
pu.cb_investments_barplot(
    ind_ts,
    y_column="raised_amount_usd_total",
    y_label="Raised amount (million USD)",
    x_label="Year",
)

# %%
pu.cb_investments_barplot(
    ind_ts,
    y_column="raised_amount_gbp_total",
    y_label="Raised amount (million GBP)",
    x_label="Year",
)

# %%
pu.cb_investments_barplot(
    ind_ts,
    y_column="no_of_rounds",
    y_label="Number of rounds",
    x_label="Year",
)

# %% [markdown]
# ## Company labels

# %%
import pandas as pd

# %%
# def fetch_company_labels(c)
label = "SUB INDUSTRIES"
sub = DR.company_subindustries.rename(columns={label: "Category"})

label = "INDUSTRIES"
ind = DR.company_industries.rename(columns={label: "Category"})

label = "TAGS"
tags = DR.company_tags.rename(columns={label: "Category"})

company_labels = pd.concat([sub, ind, tags], ignore_index=True)
company_labels = company_labels[-company_labels.Category.isnull()]


# %%
company_labels.head(3)

# %% [markdown]
# ## Country performance

# %%
country_funding = (
    DR.funding_rounds.merge(DR.company_data[["id", "country"]])
    .merge(company_labels)
    .groupby("country")
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"))
)

# %%
country_funding.sort_values("raised_amount_gbp", ascending=False).head(15)

# %%
country_funding = (
    DR.funding_rounds.merge(DR.company_data[["id", "country"]])
    .merge(company_labels)
    .groupby(["country", "Category"])
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"))
).reset_index()

# %%
country_funding.query("country == 'United Kingdom'").sort_values(
    "raised_amount_gbp", ascending=False
).head(20)

# %%
country_funding.query("country == 'United Kingdom'").query(
    "Category in @SUBINDUSTRIES"
).sort_values("raised_amount_gbp", ascending=False).head(20)

# %%
country_funding.query("country == 'United States'").query(
    "Category in @INDUSTRIES"
).sort_values("raised_amount_gbp", ascending=False).head(20)

# %%
country_funding.query("country == 'United States'").sort_values(
    "raised_amount_gbp", ascending=False
).head(20)

# %% [markdown]
# ## Embeddings
#
# - Calculate embeddings for each company
# - Calculate embeddings for each industry, sub-industry and tags
# - Calculate average embedding, and similarity between all companies

# %%
import innovation_sweet_spots.utils.embeddings_utils as eu
from innovation_sweet_spots import PROJECT_DIR
import re
import innovation_sweet_spots.utils.text_cleaning_utils as tcu

EMBEDDING_MODEL = "all-mpnet-base-v2"
EMBEDINGS_DIR = PROJECT_DIR / "outputs/preprocessed/embeddings"


# %%
def clean_text(text):
    return " ".join(
        [
            t.strip()
            for t in tcu.remove_non_alphabet(re.sub(r"\([^()]*\)", "", text)).split()
        ]
    )


# %%
company_labels_ = company_labels.copy().assign(
    Category=lambda df: df.Category.apply(clean_text)
)

# %%
company_labels_list = company_labels_.groupby("id")["Category"].apply(list)

# %%
company_labels_list.head(2)

# %%
labels_unique = list(company_labels_.Category.unique())

# %%
len(labels_unique)

# %%
importlib.reload(eu)
v_labels = eu.Vectors(
    model_name=EMBEDDING_MODEL,
    folder=EMBEDINGS_DIR,
    filename="foodtech_may2022_labels",
)

# %%
v_labels.generate_new_vectors(
    new_document_ids=labels_unique,
    texts=labels_unique,
)

# %%
v_labels.save_vectors()

# %%
importlib.reload(eu)
v_companies = eu.Vectors(
    model_name=EMBEDDING_MODEL,
    folder=EMBEDINGS_DIR,
    filename="foodtech_may2022_companies",
)

# %%
DR.company_data.TAGLINE = DR.company_data.TAGLINE.fillna("")

# %%
v_companies.generate_new_vectors(
    new_document_ids=DR.company_data.id.to_list(),
    texts=DR.company_data.TAGLINE.to_list(),
)

# %%
v_companies.save_vectors()

# %%
v_companies.vectors.shape

# %%
v_companies.select_vectors

# %%
import numpy as np

category_vectors = []
for i in v_companies.vector_ids:
    if i in company_labels_list.index.to_list():
        category_vectors.append(
            v_labels.select_vectors(company_labels_list.loc[i]).mean(axis=0)
        )
    else:
        category_vectors.append(v_companies.select_vectors([i]).mean(axis=0))


# %%
category_vectors = np.array(category_vectors)

# %%
category_vectors.shape

# %%
# company_labels_list.loc['890906']

# %%
# v_companies.vector_ids

# %%
# DR.company_data.query("id == '3029048'")

# %%
v_vectors = 0.5 * v_companies.vectors + 0.5 * category_vectors

# %%
v_vectors.shape

# %%
import umap

# %%
# Reduce the embedding to 2 dimensions
reducer = umap.UMAP(
    n_components=2,
    random_state=1,
    n_neighbors=8,
    min_dist=0.3,
    spread=0.5,
)
embedding = reducer.fit_transform(v_vectors)

# %%
from sklearn.cluster import KMeans

clusterer = KMeans(n_clusters=120, random_state=10)
clusterer.fit(embedding)
soft_clusters = list(clusterer.labels_)

# %%
len(np.unique(soft_clusters))

# %%
from nltk.corpus import stopwords

# %%
import re


def preproc(text: str) -> str:
    text = re.sub(r"[^a-zA-Z ]+", "", text).lower()
    text = text.split()
    text = [t for t in text if t not in stopwords.words("english")]
    return " ".join(text)


# %%
title_texts = DR.company_data.TAGLINE.apply(preproc)

# %%
# %%
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from innovation_sweet_spots.utils import cluster_analysis_utils

importlib.reload(cluster_analysis_utils)

cluster_texts = cluster_analysis_utils.cluster_texts(title_texts, soft_clusters)

# %%
cluster_keywords = cluster_analysis_utils.cluster_keywords(
    documents=list(cluster_texts.values()),
    cluster_labels=list(cluster_texts.keys()),
    n=3,
    max_df=0.8,
    min_df=0.01,
    Vectorizer=CountVectorizer,
)

# %%
df_viz = DR.company_data[["id", "NAME", "TAGLINE", "WEBSITE"]].copy()
df_viz["x"] = embedding[:, 0]
df_viz["y"] = embedding[:, 1]
df_viz["cluster"] = soft_clusters
df_viz["cluster_"] = [str(x) for x in soft_clusters]
centroids = (
    df_viz.groupby("cluster")
    .agg(x_c=("x", "mean"), y_c=("y", "mean"))
    .reset_index()
    .assign(
        keywords=lambda x: x.cluster.apply(lambda y: ", ".join(cluster_keywords[y]))
    )
)
df_viz = df_viz.merge(company_labels_list.reset_index(), how="left")

# %%
alt.data_transformers.disable_max_rows()

# %%
# list_of_columns = list(df_viz.columns)

# %%
# importlib.reload(pu);

# %%
# Visualise using altair
fig = (
    alt.Chart(df_viz, width=1000, height=1000)
    .mark_circle(size=20, color=pu.NESTA_COLOURS[1])
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        tooltip=list_of_columns,
        #         color="Primary Category:N",
        # color="soft_cluster_:N",
        href="WEBSITE",
        #         size="Hours per Week",
    )
)

text = (
    alt.Chart(centroids)
    .mark_text(font=pu.FONT)
    .encode(x=alt.X("x_c:Q"), y=alt.Y("y_c:Q"), text=alt.Text("keywords"))
)

fig_final = (
    (fig + text)
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
filename = "dealroom_landscape"
AltairSaver.save(fig_final, filename, filetypes=["html", "png"])

# %%
