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
# # Analysing guardian article trends

# %%
from innovation_sweet_spots.getters._foodtech import get_guardian_searches
from innovation_sweet_spots.getters.google_sheets import get_foodtech_search_terms
from innovation_sweet_spots.utils import plotting_utils as pu
import altair as alt
import pandas as pd
import numpy as np
from innovation_sweet_spots import PROJECT_DIR
import utils
import importlib
from innovation_sweet_spots.utils.pd import pd_analysis_utils as pd_au
from innovation_sweet_spots.analysis import analysis_utils as au

# %% [markdown]
# # Plotting utils

# %%
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver(path=alt_save.FIGURE_PATH + "/foodtech")

# %% [markdown]
# # Load and check search results

# %%
from innovation_sweet_spots.getters import google_sheets

importlib.reload(google_sheets)
df_id_to_term_reviewed = google_sheets.get_foodtech_guardian(from_local=False)

# %%
# Fetch list of search terms from the google sheet
df_search_terms = get_foodtech_search_terms(from_local=False).assign(
    Terms=lambda df: df.Terms.apply(utils.remove_space_after_comma)
)

# %%
df_id_to_term = (
    df_id_to_term_reviewed.copy()
    .query("checked_subcategory != '-'")
    .assign(
        **{
            "Category": lambda df: df.checked_category,
            "Sub Category": lambda df: df.checked_subcategory,
        }
    )
    .astype({"year": int})
)

# %%
df_id_to_term.year.unique()

# %%
# df_id_to_text = (
#     df_id_to_term[["id", "year", "Category", "Sub Category"]]
#     .merge(df_id_to_term_copy[["id", "text"]], on="id", how="left")
#     .drop_duplicates(["id", "Sub Category"])
#     .fillna("")
# )

# %%
# df_id_to_text.query('`Sub Category`=="Delivery"').to_csv(
#     PROJECT_DIR
#     / "outputs/foodtech/interim/public_discourse/guardian_articles_delivery.csv",
#     index=False,
# )
# df_id_to_text.query('`Sub Category`=="Reformulation"').to_csv(
#     PROJECT_DIR
#     / "outputs/foodtech/interim/public_discourse/guardian_articles_reformulation.csv",
#     index=False,
# )

# %%
(
    df_id_to_term.query("year > 2016 and year < 2022")
    .groupby(["Category", "Sub Category", "Tech area"], as_index=False)
    .agg(counts=("id", "count"))
)

# %%
len(
    df_id_to_term.query("year > 2016 and year < 2022")
    .query("`Sub Category` == 'Alt protein'")
    .drop_duplicates("id")
)

# %%
105 / 169

# %%
43 / 169

# %%
6 / 169

# %% [markdown]
# # Checking technology trends
# - Combining terms by categories
# - Time series with terms and categories
# - Calculate magnitude and growth

# %%
from innovation_sweet_spots import PROJECT_DIR

# Total article counts per year
guardian_baseline = pd.read_csv(
    PROJECT_DIR / "outputs/foodtech/interim/public_discourse/guardian_baseline.csv"
)


def get_ts(df_id_to_term, category="Category"):
    """build time series"""
    return (
        df_id_to_term.drop_duplicates(["id", category], keep="first")
        .groupby(["year", category])
        .agg(counts=("id", "count"))
        .reset_index()
        .merge(
            guardian_baseline.rename(columns={"counts": "total_counts"}),
            on="year",
            how="left",
        )
        .assign(fraction=lambda df: df.counts / df.total_counts)
    )


# %%
ts_category = get_ts(df_id_to_term, "Category")

# %%
ts_subcategory = get_ts(df_id_to_term, "Sub Category").merge(
    df_search_terms[["Category", "Sub Category"]].drop_duplicates("Sub Category"),
    how="left",
)

# %%
ts_tech_area = get_ts(df_id_to_term, "Tech area").merge(
    df_search_terms[["Category", "Sub Category", "Tech area"]].drop_duplicates(
        "Tech area"
    ),
    how="left",
)

# %%
# ts_search_term = get_ts(df_id_to_term, "Terms").merge(
#     df_search_terms[["Category", "Sub Category", "Tech area", "Terms"]].drop_duplicates(
#         "Terms"
#     ),
#     how="left",
# )

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_category)
    .mark_line()
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale)),
        y=alt.Y("fraction:Q", sort="-x"),
        # size=alt.Size('magnitude'),
        color="Category",
        tooltip=["year", "counts", "Category"],
    )
)
fig

# %%
alt.Chart(ts_subcategory, width=200, height=100).mark_line(size=3).encode(
    x="year:O",
    y=alt.Y(
        "counts:Q",
    ),
    color="Sub Category:N",
    facet=alt.Facet("Category:N", columns=2),
    tooltip=["Sub Category", "counts", "year"],
).resolve_scale(y="independent")

# %% [markdown]
# ## Magnitude and growth trends

# %%
from innovation_sweet_spots.utils import chart_trends

# %%
categories_to_check = ts_category.Category.unique()
variable = "fraction"
magnitude_growth = []
for tech_area in categories_to_check:
    print(tech_area)
    df = ts_category.query("Category == @tech_area").drop("Category", axis=1)[
        ["year", variable]
    ]
    df_trends = au.estimate_magnitude_growth(df, 2017, 2021)
    magnitude_growth.append(
        [
            df_trends.query('trend == "magnitude"').iloc[0][variable],
            df_trends.query('trend == "growth"').iloc[0][variable],
            tech_area,
        ]
    )
magnitude_growth_df = pd.DataFrame(
    magnitude_growth, columns=["magnitude", "growth", "tech_area"]
).assign(
    growth=lambda df: df.growth / 100,
    magnitude=lambda df: df.magnitude * 100,
)

# %%
domain = [
    "Health",
    "Innovative food",
    "Logistics",
    "Restaurants and retail",
    "Cooking and kitchen",
    "Food waste",
]
range_ = pu.NESTA_COLOURS[0 : len(domain)]

# %%
magnitude_growth_df

# %%
baseline_magnitude_growth = au.estimate_magnitude_growth(guardian_baseline, 2017, 2021)
baseline_growth = (
    baseline_magnitude_growth.query("trend == 'growth'").iloc[0].counts / 100
)

# %%
baseline_growth

# %%
magnitude_growth_df.magnitude.median()

# %%
fig = chart_trends.mangitude_vs_growth_chart(
    data=magnitude_growth_df,
    x_limit=0.45,
    y_limit=4,
    mid_point=0.09,
    baseline_growth=0,
    values_label="Proportion of articles (%)",
    text_column="tech_area",
)
fig

# %%
# fig = chart_trends.mangitude_vs_growth_chart(
#     data=magnitude_growth_df,
#     x_limit=0.2,
#     y_limit=4,
#     mid_point=0.053,
#     baseline_growth=0,
#     values_label="Proportion of articles (%)",
#     text_column="tech_area",
# )
# fig

# %%
variable = "counts"
magnitude_growth = []
for tech_area in categories_to_check:
    df = ts_category.query("Category == @tech_area").drop("Category", axis=1)[
        ["year", variable]
    ]
    df_trends = au.estimate_magnitude_growth(df, 2017, 2021)
    magnitude_growth.append(
        [
            df_trends.query('trend == "magnitude"').iloc[0][variable],
            df_trends.query('trend == "growth"').iloc[0][variable],
            tech_area,
        ]
    )
magnitude_growth_df = pd.DataFrame(
    magnitude_growth, columns=["magnitude", "growth", "tech_area"]
).assign(
    growth=lambda df: df.growth / 100,
)

# %%
magnitude_growth_df.magnitude.median()

# %%
fig = chart_trends.mangitude_vs_growth_chart(
    data=magnitude_growth_df,
    x_limit=400,
    y_limit=3,
    mid_point=72,
    baseline_growth=-0.25,
    values_label="Average number of articles",
    text_column="tech_area",
)
fig

# %%
(
    alt.Chart(magnitude_growth_df)
    .mark_bar()
    .encode(
        x=alt.X("growth"),
        y=alt.Y("tech_area"),
    )
)

# %%
AltairSaver.save(
    fig_final, f"Guardian_articles_magnitude_growth", filetypes=["html", "svg", "png"]
)

# %%
ts_subcategory.head(1)

# %%
cats = ["Reformulation", "Alt protein"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df.query("year > 2010")).mark_line(size=3, interpolate="monotone")
    # .mark_bar(size=5, interpolate="monotone")
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y("counts:Q", title="Number of articles"),
        color=alt.Color("Sub Category:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        # tooltip=["year", "counts", "query", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig, f"Guardian_articles_per_year_InnovativeFood", filetypes=["html", "svg", "png"]
)

# %%
cats = ["Delivery"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df)
    .mark_line(size=3, interpolate="monotone", color=pu.NESTA_COLOURS[3])
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y("counts:Q", title="Number of articles"),
        color=alt.Color("Sub Category:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        # tooltip=["year", "counts", "query", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig, f"Guardian_articles_per_year_Delivery", filetypes=["html", "svg", "png"]
)

# %%
cats = ["Restaurants", "Retail"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df).mark_line(
        size=3, interpolate="monotone", color=pu.NESTA_COLOURS[3]
    )
    # .mark_bar(size=5, interpolate="monotone", color=pu.NESTA_COLOURS[3])
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y("counts:Q", title="Number of articles"),
        color=alt.Color("Sub Category:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        # tooltip=["year", "counts", "query", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
cats = ["Kitchen tech"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df)
    .mark_line(size=3, interpolate="monotone", color=pu.NESTA_COLOURS[3])
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y("counts:Q", title="Number of articles"),
        color=alt.Color("Sub Category:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        # tooltip=["year", "counts", "query", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
cats = ["Food waste"]
ts_df = ts_subcategory.query("`Sub Category` in @cats")

# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df)
    .mark_line(size=3, interpolate="monotone", color=pu.NESTA_COLOURS[3])
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y("fraction:Q", title="Number of articles"),
        color=alt.Color("Sub Category:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
categories_to_check

# %%
categories_to_check = ts_subcategory["Sub Category"].unique()
variable = "fraction"
magnitude_growth = []
for tech_area in categories_to_check:
    print(tech_area)
    df = ts_subcategory.query("`Sub Category` == @tech_area").drop("Category", axis=1)[
        ["year", variable]
    ]
    df_trends = au.estimate_magnitude_growth(df, 2017, 2021)
    magnitude_growth.append(
        [
            df_trends.query('trend == "magnitude"').iloc[0][variable],
            df_trends.query('trend == "growth"').iloc[0][variable],
            tech_area,
        ]
    )
magnitude_growth_df = pd.DataFrame(
    magnitude_growth, columns=["magnitude", "growth", "tech_area"]
).assign(
    growth=lambda df: df.growth / 100,
    magnitude=lambda df: df.magnitude * 100,
)

# %%
magnitude_growth_df

# %%
categories_to_check = ts_tech_area["Tech area"].unique()
variable = "fraction"
magnitude_growth = []
for tech_area in categories_to_check:
    print(tech_area)
    try:
        df = ts_tech_area.query("`Tech area` == @tech_area").drop("Category", axis=1)[
            ["year", variable]
        ]
        df_trends = au.estimate_magnitude_growth(df, 2017, 2021)
        magnitude_growth.append(
            [
                df_trends.query('trend == "magnitude"').iloc[0][variable],
                df_trends.query('trend == "growth"').iloc[0][variable],
                tech_area,
            ]
        )
    except:
        pass
magnitude_growth_df = pd.DataFrame(
    magnitude_growth, columns=["magnitude", "growth", "tech_area"]
).assign(
    growth=lambda df: df.growth / 100,
    magnitude=lambda df: df.magnitude * 100,
)

# %%
magnitude_growth_df.sort_values("magnitude")

# %% [markdown]
# g.document_mentions## Checking obesity/health trends

# %%
from innovation_sweet_spots.utils.pd import pd_analysis_utils as pdau

banned_terms = ["Australia"]


def get_g_ts(g):
    """Get time series from a search query"""
    return (
        g.document_mentions.rename(columns={"documents": "counts"})
        .merge(
            guardian_baseline.rename(columns={"counts": "total_counts"}),
            on="year",
            how="left",
        )
        .assign(fraction=lambda df: df.counts / df.total_counts)
    )


def get_queries_ts(queries, required_terms=None):
    """Get time series for all specified search queries"""
    ts_df = []
    for query_id in queries:
        required_terms = queries[query_id] if required_terms is None else required_terms
        # REQUIRED_TERMS = search_terms
        g = pdau.DiscourseAnalysis(
            search_terms=queries[query_id],
            required_terms=required_terms,
            banned_terms=banned_terms,
            use_cached=True,
            query_identifier=query_id,
        )
        ts_df.append(get_g_ts(g).assign(query=query_id))
    ts_df = pd.concat(ts_df, ignore_index=True)
    return ts_df


# %%
# {query_id: search terms}
queries = {
    "obesity": ["obesity", "obese"],
    "overweight": ["overweight"],
    "healthy_eating": [
        "healthy food",
        "healthy foods",
        "healthy eating",
        "healthy meal",
        "healthy meals",
    ],
    "food_environment": ["food environment", "food environments", "obesogenic"],
    # "food_desert": ["food desert"],
}


ts_df = get_queries_ts(queries)


# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df)
    .mark_line(size=3)
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale)),
        y=alt.Y("fraction:Q", sort="-x"),
        color=alt.Color("query:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts", "query", alt.Tooltip("fraction:Q", format="%")],
    )
)
fig

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df)
    .mark_line(size=3)
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale)),
        y=alt.Y("counts:Q", sort="-x"),
        color=alt.Color("query:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts", "query"],
    )
)
fig

# %%
ts_df_obesity = ts_df.query("query=='obesity'")
# ts_df_obesity = ts_df.query("query=='food_environment'")

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df_obesity)
    .mark_line(size=3, color=pu.NESTA_COLOURS[0], interpolate="monotone")
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y("counts:Q", title="Number of articles"),
        # color=alt.Color("query:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts", "query"],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig, f"Guardian_articles_per_year_Obesity", filetypes=["html", "svg", "png"]
)

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df_obesity)
    .mark_line(size=3, color=pu.NESTA_COLOURS[0], interpolate="monotone")
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y(
            "fraction:Q",
            title="Percentage of articles",
            axis=alt.Axis(
                format="%",
                tickCount=5,
            ),
        ),
        # color=alt.Color("query:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts", "query", alt.Tooltip("fraction:Q", format=".2%")],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig, f"Guardian_proportion_per_year_Obesity", filetypes=["html", "svg", "png"]
)

# %%
au.estimate_magnitude_growth(ts_df_obesity, 2017, 2021)

# %%
ts_df_environment = ts_df.query("query=='food_environment'")

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(ts_df_environment)
    .mark_line(size=3, color=pu.NESTA_COLOURS[0], interpolate="monotone")
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale), title=""),
        y=alt.Y("counts:Q", title="Number of articles"),
        # color=alt.Color("query:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "counts", "query"],
    )
)
fig = pu.configure_plots(fig)
fig

# %%
au.estimate_magnitude_growth(ts_df_environment, 2017, 2021)

# %% [markdown]
# ### Overlap of obesity and food tech articles

# %%
query_id = "obesity"
g = pdau.DiscourseAnalysis(
    search_terms=queries[query_id],
    required_terms=queries[query_id],
    banned_terms=banned_terms,
    use_cached=True,
    query_identifier=query_id,
)

# %%
obesity_ids = list(g.metadata.keys())
years = [int(g.metadata[key]["webPublicationDate"][0:4]) for key in g.metadata]
obesity_ids = set(
    [x for i, x in enumerate(obesity_ids) if ((years[i] < 2022) and (years[i] >= 2017))]
)


# %%
len(obesity_ids)

# %%
foodtech_ids = set(df_id_to_term.query("year >= 2017 and year < 2022").id.to_list())

# %%
common_articles = len(obesity_ids.intersection(foodtech_ids))

# %%
common_articles / len(obesity_ids)

# %%
common_articles / len(foodtech_ids)

# %%
for cat in df_id_to_term.Category.unique():
    foodtech_ids = set(
        df_id_to_term.query("year >= 2017 and year < 2022")
        .query("Category==@cat")
        .id.to_list()
    )
    common_articles = len(obesity_ids.intersection(foodtech_ids))
    print(cat)
    print(common_articles / len(foodtech_ids))

# %%
query_id = "healthy_eating"
g = pdau.DiscourseAnalysis(
    search_terms=queries[query_id],
    required_terms=[],
    banned_terms=banned_terms,
    use_cached=True,
    query_identifier=query_id,
)

# %%
ids = set(list(g.metadata.keys()))

# %%
for cat in df_id_to_term.Category.unique():
    foodtech_ids = set(df_id_to_term.query("Category==@cat").id.to_list())
    common_articles = len(ids.intersection(foodtech_ids))
    print(cat)
    print(common_articles / len(foodtech_ids))

# %% [markdown]
# ## Check mentions of Obesity AND food environment related terms

# %%
food_environment_terms = [
    "food environment",
    "advertising",
    "marketing",
    "advert",
    "takeaway service",
    "meal takeaway",
    "food takeaway",
    "fast food",
    "junk food",
    "ultra processed food",
    "food desert",
    "food system",
    "inequality",
    "inequalities",
    "poverty",
    "obesogenic",
]

# %%
ts_df = get_queries_ts({"obesity": queries["obesity"]})

# %%
ts_df_ = get_queries_ts(
    {"obesity": queries["obesity"]}, required_terms=food_environment_terms
)

# %%
ts_df_["query"] = "obesity_x_food_env"

# %%
ts_df_combined = pd.concat([ts_df, ts_df_])


# %%
df = ts_df.merge(ts_df_[["counts", "year"]], on="year").assign(
    fraction_environment=lambda df: df.counts_y / df.counts_x
)

# %%
# scale = 'log'
scale = "linear"

fig = (
    alt.Chart(df)
    .mark_line(size=3)
    .encode(
        x=alt.X("year:O", scale=alt.Scale(type=scale)),
        y=alt.Y("fraction_environment:Q", sort="-x"),
        color=alt.Color("query:N"),
        # size=alt.Size('magnitude'),
        # color='Category',
        tooltip=["year", "fraction_environment", "query"],
    )
)
fig

# %%
