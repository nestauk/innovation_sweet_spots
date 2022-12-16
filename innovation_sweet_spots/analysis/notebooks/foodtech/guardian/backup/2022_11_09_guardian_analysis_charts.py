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

# %% [markdown]
# # Plotting utils

# %%
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver(path=alt_save.FIGURE_PATH + "/foodtech")


# %% [markdown]
# # Load and check search results

# %%
def remove_space_after_comma(text):
    """util function to process search terms with comma"""
    return ",".join([s.strip() for s in text.split(",")])


# %%
# Fetch search terms
df_search_terms = get_foodtech_search_terms(from_local=False)
df_search_terms["Terms"] = df_search_terms["Terms"].apply(remove_space_after_comma)

# %%
df_search_results_path = (
    PROJECT_DIR / "outputs/foodtech/interim/public_discourse/Food_terms_table_V2.csv"
)
df_precision_path = (
    PROJECT_DIR / "outputs/foodtech/interim/public_discourse/Food_terms_precision.csv"
)
df_search_results = pd.read_csv(df_search_results_path)
df_precision = pd.read_csv(df_precision_path)

# %%
imprecise_terms = df_precision.query("proportion_correct <= 0.5").terms.to_list()

# %%
# Get Guardian search results
# df_search_results = get_guardian_searches()

# %%
df_search_results.head(1)

# %%
df_precision.head(1)

# %%
# Get article counts for each term
cols_to_drop = ["id", "text", "date", "year", "URL", "Headline", "Unnamed: 0"]

df_counts = (
    pd.DataFrame(df_search_results.drop(cols_to_drop, axis=1).sum())
    .rename(columns={0: "counts"})
    .sort_values("counts", ascending=False)
    .reset_index()
    .rename(columns={"index": "Terms"})
    .merge(
        df_search_terms[["Category", "Sub Category", "Tech area", "Terms"]], how="left"
    )
)

# %%
# Terms without any hits
# df_counts[df_counts.counts == 0].sort_values(["Sub Category", "Tech area"])

# %%
# Terms without any hits
# df_counts[df_counts.counts > 0].sort_values(["Sub Category", "Tech area"]).iloc[101:]

# %%
terms_to_remove = ["supply chain"] + imprecise_terms

# %%
assert df_search_results.id.duplicated().sum() == 0

# %%
# Link articles to categories, sub categories and tech areas
non_search_term_columns = ["year", "text", "date", "URL", "Headline", "Unnamed: 0"]
df_id_to_term = df_search_results.drop(
    set(non_search_term_columns).difference({"year"}), axis=1
).copy()

df_id_to_term = (
    pd.melt(df_id_to_term, id_vars=["id", "year"])
    .query("value==1")
    .rename(columns={"variable": "Terms"})
    .drop("value", axis=1)
    .merge(
        df_search_terms[["Category", "Sub Category", "Tech area", "Terms"]], how="left"
    )
)


# %%
# # Remove articles with imprecise terms
# df_id_to_term = df_id_to_term[
#     df_id_to_term.Terms.isin(terms_to_remove) == False
# ].reset_index(drop=True)

# %%
n_terms_per_article = df_id_to_term.groupby("id", as_index=False).agg(
    counts=("Terms", "count")
)
article_has_imprecise_term = (
    df_id_to_term.copy()
    .assign(has_imprecise_term=lambda df: df.Terms.isin(terms_to_remove))
    .merge(n_terms_per_article, on="id", how="left")
)

# %%
articles_to_keep = article_has_imprecise_term.assign(
    keep=lambda df: (df.has_imprecise_term == False) | (df.counts > 1)
)

# %%
articles_to_keep.keep.sum()

# %%
df_id_to_term = articles_to_keep[articles_to_keep.keep == True].copy()

# %%
len(df_id_to_term)

# %%
new_terms = list(df_id_to_term[df_id_to_term.Category.isnull()].Terms.unique())

# %%
df_id_to_term.loc[df_id_to_term.Terms.isin(new_terms), "Category"] = "Innovative food"
df_id_to_term.loc[df_id_to_term.Terms.isin(new_terms), "Sub Category"] = "Reformulation"
df_id_to_term.loc[df_id_to_term.Terms.isin(new_terms), "Tech area"] = "Reformulation"

# %%
# Remove articles containing Australia in the title
df_id_to_term = df_id_to_term[df_id_to_term.id.str.contains("australia") == False]
df_id_to_term = df_id_to_term[df_id_to_term.id.str.contains("Australia") == False]

# %%
## Check terms that made the final selection

# %%
counts_df = (
    df_id_to_term.drop_duplicates(["Terms", "id"])
    .groupby(["Terms", "Category", "Sub Category"], as_index=False)
    .agg(counts=("id", "count"))
)

# %%
counts_df.sort_values(["Category", "Sub Category", "Terms"]).iloc[51:]

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
ts_search_term = get_ts(df_id_to_term, "Terms").merge(
    df_search_terms[["Category", "Sub Category", "Tech area", "Terms"]].drop_duplicates(
        "Terms"
    ),
    how="left",
)

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

# %%
alt.Chart(ts_tech_area, width=200, height=100).mark_line().encode(
    x="year:O",
    y=alt.Y(
        "fraction:Q",
    ),
    color=alt.Color("Tech area:N", scale=alt.Scale(scheme="dark2")),
    facet=alt.Facet("Category:N", columns=2),
    tooltip=["Tech area", "counts", "year"],
).resolve_scale(y="independent")

# %%
alt.Chart(ts_search_term, width=200, height=100).mark_line(
    point=alt.OverlayMarkDef()
).encode(
    x="year:O",
    y=alt.Y(
        "fraction:Q",
    ),
    color=alt.Color("Terms:N", scale=alt.Scale(scheme="dark2")),
    facet=alt.Facet("Category:N", columns=2),
    tooltip=["Terms", "counts"],
).resolve_scale(
    y="independent"
)

# %%
alt.Chart(ts_search_term, width=200, height=100).mark_line().encode(
    x="year:O",
    y=alt.Y(
        "fraction:Q",
    ),
    color=alt.Color("Terms:N", scale=alt.Scale(scheme="dark2")),
    facet=alt.Facet("Category:N", columns=2),
    tooltip=["Terms", "counts"],
).resolve_scale(y="independent")

# %%
import importlib
from innovation_sweet_spots.analysis import analysis_utils as au

importlib.reload(au)

# %%
# Export articles to check

# %%
df_to_check = df_id_to_term.sort_values(
    ["year", "Tech area", "Sub Category", "Category"]
).merge(df_search_results[["id", "Headline", "URL"]])

# %%
df_to_check.to_csv(
    PROJECT_DIR / "outputs/foodtech/interim/public_discourse/guardian_to_check_V2.csv"
)

# %%
len(df_to_check)

# %% [markdown]
# ## Magnitude and growth trends

# %%
from innovation_sweet_spots.utils import chart_trends

# %%
categories_to_check = ts_category.Category.unique()

# %%
magnitude_growth = []
for tech_area in categories_to_check:
    df = ts_category.query("Category == @tech_area").drop("Category", axis=1)[
        ["year", "counts"]
    ]
    df_trends = au.estimate_magnitude_growth(df, 2017, 2021)
    magnitude_growth.append(
        [
            df_trends.query('trend == "magnitude"').iloc[0].counts,
            df_trends.query('trend == "growth"').iloc[0].counts,
            tech_area,
        ]
    )
magnitude_growth_df = pd.DataFrame(
    magnitude_growth, columns=["magnitude", "growth", "tech_area"]
).assign(growth=lambda df: df.growth / 100)

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
# make the plot...
import altair as alt
from innovation_sweet_spots.utils import plotting_utils as pu

colour_field = "tech_area"
text_field = "tech_area"
horizontal_scale = "linear"
# horizontal_scale = "log"
horizontal_title = f"Average number of articles"
legend = alt.Legend()

title_text = "News article trends (2017-2021)"
subtitle_text = [
    # "Data: Dealroom. Showing data on early stage deals (eg, seed and series funding)",
    # "Late stage deals, such as IPOs, acquisitions, and debt financing not included.",
]

fig = (
    alt.Chart(
        magnitude_growth_df,
        width=400,
        height=400,
    )
    .mark_circle(size=80)
    .encode(
        x=alt.X(
            "magnitude:Q",
            axis=alt.Axis(
                title=horizontal_title,
                tickCount=5,
            ),
            scale=alt.Scale(
                type=horizontal_scale,
                # domain=(0, 40),
            ),
        ),
        y=alt.Y(
            "growth:Q",
            axis=alt.Axis(
                title="Growth",
                format="%",
                tickCount=5,
            ),
            scale=alt.Scale(
                # domain=(-1, 2.5),
            ),
        ),
        color=alt.Color(
            f"{colour_field}:N",
            legend=None,
            scale=alt.Scale(domain=domain, range=range_),
        ),
        tooltip=[
            alt.Tooltip("tech_area", title="Category"),
            alt.Tooltip("magnitude", title=horizontal_title),
            alt.Tooltip("growth", title="Growth", format=".0%"),
        ],
    )
    .properties(
        title={
            "anchor": "start",
            "text": title_text,
            "subtitle": subtitle_text,
            "subtitleFont": pu.FONT,
            "fontSize": 15,
        },
    )
)

text = fig.mark_text(
    align="left", baseline="middle", font=pu.FONT, dx=7, fontSize=15
).encode(text=text_field)

yrule = (
    alt.Chart(pd.DataFrame({"y": [baseline_growth]}))
    .mark_rule(strokeDash=[5, 7], size=1)
    .encode(y="y:Q")
)

fig_final = (
    (fig + yrule + text)
    .configure_axis(
        grid=False,
        gridDash=[5, 7],
        # gridColor="grey",
        labelFontSize=pu.FONTSIZE_NORMAL,
        titleFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_legend(
        titleFontSize=pu.FONTSIZE_NORMAL,
        labelFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_view(strokeWidth=0)
)

fig_final

# %%
magnitude_growth_df.magnitude.median()

# %%
fig = chart_trends.mangitude_vs_growth_chart(
    data=magnitude_growth_df,
    x_limit=300,
    y_limit=1.50,
    mid_point=108,
    baseline_growth=-0.25,
    values_label="Average number of articles",
    text_column="tech_area",
)
fig

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
    alt.Chart(ts_df)
    .mark_line(size=3, interpolate="monotone")
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
cats = ["Delivery", "Supply chain"]
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
AltairSaver.save(
    fig, f"Guardian_articles_per_year_Delivery", filetypes=["html", "svg", "png"]
)

# %%
cats = ["Restaurants and retail"]
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
