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
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Producing graphs for blogs

# %%
from innovation_sweet_spots import PROJECT_DIR, logging, config
from innovation_sweet_spots.getters import gtr, crunchbase, guardian
import innovation_sweet_spots.analysis.analysis_utils as iss
import innovation_sweet_spots.analysis.topic_analysis as iss_topics

# %%
import pandas as pd
import numpy as np
import altair as alt

from matplotlib import pyplot as plt
import seaborn as sns

fontsize_med = 12
plt.rcParams["svg.fonttype"] = "none"

# %%
import innovation_sweet_spots.utils.io as iss_io

# %%
# Import Crunchbase data
crunchbase.CB_PATH = crunchbase.CB_PATH.parent / "cb_2021"
cb = crunchbase.get_crunchbase_orgs_full()
cb_df = cb[-cb.id.duplicated()]
cb_df = cb_df[cb_df.country == "United Kingdom"]
cb_df = cb_df.reset_index(drop=True)
del cb
cb_investors = crunchbase.get_crunchbase_investors()
cb_investments = crunchbase.get_crunchbase_investments()
cb_funding_rounds = crunchbase.get_crunchbase_funding_rounds()

# %%
gtr_projects = gtr.get_gtr_projects()

# %% [markdown]
# ## Functions for graphs

# %%
PLOT_LCH_CATEGORY_DOMAIN = [
    "Heat pumps",
    "Biomass heating",
    "Hydrogen heating",
    "Geothermal energy",
    "Solar thermal",
    "District heating",
    "Heat storage",
    "Insulation & retrofit",
    "Energy management",
    "Micro CHP",
]

PLOT_LCH_CATEGORY_COLOURS = [
    "#4c78a8",
    "#f58518",
    "#e45756",
    "#72b7b2",
    "#54a24b",
    "#eeca3b",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
]
COLOR_PAL_LCH = (PLOT_LCH_CATEGORY_DOMAIN, PLOT_LCH_CATEGORY_COLOURS)

PLOT_REF_CATEGORY_DOMAIN = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "LCH & EEM",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
]

PLOT_REF_CATEGORY_COLOURS = [
    "#4c78a8",
    "#f58518",
    "#e45756",
    "#72b7b2",
    "#54a24b",
    "#eeca3b",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    #     '#bab0ac',
]

COLOR_PAL_REF = (PLOT_REF_CATEGORY_DOMAIN, PLOT_REF_CATEGORY_COLOURS)

# %%
dark_purple = "#2F1847"


# %%
def get_year_by_year_stats(yearly_stats, variable, year_dif=4):
    df_stats_all = pd.DataFrame()
    for cat in list(yearly_stats.keys()):
        for y in range(2011, 2021):
            yy = year_dif
            #         if y==2009: yy=1;
            #         if y==2010: yy=2;
            #         if y==2011: yy=2;
            #         if y>2010: yy=4;
            df_stats = pd.DataFrame(
                [
                    get_growth_and_level(yearly_stats, cat, variable, y - yy, y)
                    + tuple([cat, y])
                    for cat in [cat]
                ],
                columns=["growth", variable, "tech_category", "year"],
            )
            df_stats_all = df_stats_all.append(df_stats, ignore_index=True)
    #     df_stats_all.loc[df_stats_all.tech_category==cat, variable] = df_stats_all.loc[df_stats_all.tech_category==cat, variable] / df_stats_all[(df_stats_all.tech_category==cat)].iloc[0][variable]
    return df_stats_all


# %%
def plot_bars(
    YEARLY_STATS, variable, cat, y_label, final_year=2020, bar_color=dark_purple
):
    """ """
    df_ = YEARLY_STATS[cat]
    df_ = df_[df_.year <= final_year]
    ww = 350
    hh = 150
    base = alt.Chart(df_, width=ww, height=hh).encode(
        alt.X(
            "year:O",
            #             axis=alt.Axis(title=None, labels=False)
        )
    )
    fig_projects = base.mark_bar(color=bar_color).encode(
        alt.Y(variable, axis=alt.Axis(title=y_label, titleColor=bar_color))
    )
    return iss.nicer_axis(fig_projects)


# %%
def plot_matrix(
    YEARLY_STATS,
    variable,
    category_names,
    x_label,
    y_label="Growth",
    year_1=2016,
    year_2=2020,
    color_pal=(PLOT_LCH_CATEGORY_DOMAIN, PLOT_LCH_CATEGORY_COLOURS),
    window=3,
):

    df_stats = pd.DataFrame(
        [
            get_growth_and_level(
                YEARLY_STATS, cat, variable, year_1, year_2, window=window
            )
            + tuple([cat])
            for cat in category_names
        ],
        columns=["growth", variable, "tech_category"],
    )

    points = (
        alt.Chart(
            df_stats,
            height=350,
            width=350,
        )
        .mark_circle(size=35)
        .encode(
            alt.Y("growth:Q", title=y_label),
            alt.X(f"{variable}:Q", title=x_label),
            color=alt.Color(
                "tech_category",
                scale=alt.Scale(domain=color_pal[0], range=color_pal[1]),
                #             color=alt.Color(
                #                 'tech_category',
                #                 scale=alt.Scale(domain=category_names,
                #                                 range=COLOUR_PAL[0:len(category_names)]),
                legend=None,
            ),
        )
    )

    text = points.mark_text(align="left", baseline="middle", dx=7, size=15).encode(
        text="tech_category"
    )

    fig = (
        (points + text)
        .configure_axis(grid=True)
        .configure_view(strokeOpacity=1, strokeWidth=2)
    )
    return fig


# %%
def plot_matrix_trajectories(
    df_stats_all,
    variable,
    category_names,
    x_label,
    y_label,
    ww=350,
    hh=350,
    color_pal=(PLOT_LCH_CATEGORY_DOMAIN, PLOT_LCH_CATEGORY_COLOURS),
):
    points = (
        alt.Chart(
            df_stats_all[df_stats_all.tech_category.isin(cats)],
            height=hh,
            width=ww,
        )
        .mark_line(size=3)
        .encode(
            alt.Y("growth:Q", title=y_label),
            alt.X(f"{variable}:Q", title=x_label),
            order="year",
            #             color=alt.Color('tech_category'),
            color=alt.Color(
                "tech_category",
                scale=alt.Scale(domain=color_pal[0], range=color_pal[1]),
            ),
            #             color=alt.Color(
            #                 'tech_category',
            #                 scale=alt.Scale(domain=category_names, range=COLOUR_PAL[0:len(category_names)]),
            #                 legend=None
            #             ),
        )
    )

    text = points.mark_text(align="left", baseline="middle", dx=4, size=9).encode(
        text="year"
    )

    fig = (
        (points + text)
        .configure_axis(grid=True)
        .configure_view(strokeOpacity=1, strokeWidth=2)
    )
    return fig


# %%
def get_growth_and_level(
    YEARLY_STATS, cat, variable, year_1=2016, year_2=2020, window=3
):
    df = YEARLY_STATS[cat].copy()
    df = df[df.year <= 2020]
    df_ma = iss_topics.get_moving_average(df, window=window, rename_cols=False)
    df = df.set_index("year")
    df_ma = df_ma.set_index("year")
    if df_ma.loc[year_1, variable] != 0:
        growth_rate = df_ma.loc[year_2, variable] / df_ma.loc[year_1, variable]
    else:
        growth_rate = np.nan
    level = df.loc[year_1:year_2, variable].mean()
    return growth_rate, level


# %%
import innovation_sweet_spots.utils.altair_save_utils as alt_save

driver = alt_save.google_chrome_driver_setup()

# %%
cb_funding_rounds = crunchbase.get_crunchbase_funding_rounds()

# %% [markdown]
# ## Import the data

# %%
YEARLY_STATS = iss_io.load_pickle(
    PROJECT_DIR
    / "outputs/data/results_august/FINAL_TABLES_yearly_stats_all_categories_2021_Funds.p"
)
GTR_DOCS_ALL_ = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/FINAL_TABLES_GTR.csv"
)
CB_DOCS_ALL_ = pd.read_csv(
    PROJECT_DIR / "outputs/data/results_august/FINAL_TABLES_CB.csv"
)

# %%
YEARLY_STATS["Hydrogen heating"]

# %%
# GTR_DOCS_ALL_[GTR_DOCS_ALL_.tech_category=='Hydrogen & fuel cells']

# %%
list(YEARLY_STATS.keys())

# %%
cb_all_yearly_funding = iss.get_cb_funding_per_year(
    iss.get_cb_org_funding_rounds(cb_df, cb_funding_rounds), max_year=2021
)

# %%
for key in YEARLY_STATS:
    YEARLY_STATS[key]["no_of_rounds_norm"] = (
        YEARLY_STATS[key]["no_of_rounds"] / cb_all_yearly_funding["no_of_rounds"]
    )
    YEARLY_STATS[key]["raised_amount_gbp_total_norm"] = (
        YEARLY_STATS[key]["raised_amount_gbp_total"]
        / cb_all_yearly_funding["raised_amount_gbp_total"]
    )


# %%
df_all_yearly_stats = pd.DataFrame()
for key in YEARLY_STATS:
    YEARLY_STATS[key]
    df_ = YEARLY_STATS[key].copy()
    df_["tech_category"] = key
    df_all_yearly_stats = df_all_yearly_stats.append(df_, ignore_index=True)

df_all_yearly_stats_norm = pd.DataFrame()
for key in YEARLY_STATS:
    YEARLY_STATS_NORM = iss_topics.get_moving_average(
        YEARLY_STATS[key], window=3, rename_cols=False
    )
    df_ = YEARLY_STATS_NORM.copy()
    df_["tech_category"] = key
    df_all_yearly_stats_norm = df_all_yearly_stats_norm.append(df_, ignore_index=True)

# %%
import innovation_sweet_spots.analysis.figures as iss_figures
import importlib

# %% [markdown]
# # Research trends

# %%
importlib.reload(iss_figures)

# %% [markdown]
# ## Barplots

# %%
cat = "Heat pumps"
fig = iss_figures.plot_bars(YEARLY_STATS, "no_of_projects", cat, "Number of projects")
alt_save.save_altair(fig, "blog_fig_Timeseries_Heat_pumps", driver)
fig

# %%
cats = sorted(
    [
        "Heat pumps",
        "Biomass heating",
        "Hydrogen heating",
        "Geothermal energy",
        "Solar thermal",
        "District heating",
        "Heat storage",
        "Insulation & retrofit",
        "Energy management",
    ]
)
for cat in cats:
    fig = iss_figures.plot_bars(
        YEARLY_STATS, "no_of_projects", cat, "Number of projects"
    )
    alt_save.save_altair(fig, f"blog_fig_Timeseries_{cat}", driver)

# %%
cat = "District heating"
fig = iss_figures.plot_bars(YEARLY_STATS, "no_of_projects", cat, "Number of projects")
fig

# %%
cat = "Low carbon heating"
fig = iss_figures.plot_bars(YEARLY_STATS, "no_of_projects", cat, "Number of projects")
fig

# %%
cat = "Low carbon heating"
fig = iss_figures.plot_bars(YEARLY_STATS, "amount_total", cat, "Number of projects")
fig

# %%
cat = "Energy efficiency & management"
fig = iss_figures.plot_bars(YEARLY_STATS, "amount_total", cat, "Number of projects")
fig

# %%
cat = "Insulation & retrofit"
iss_figures.plot_bars(YEARLY_STATS, "no_of_projects", cat, "Number of projects")

# %%
df = YEARLY_STATS["Insulation & retrofit"]
print(df[df.year.isin(list(range(2016, 2018)))].amount_total.mean())
print(df[df.year.isin(list(range(2018, 2021)))].amount_total.mean())

# %%
iss_topics.get_moving_average(df, window=3, rename_cols=False)[
    ["year", "no_of_projects"]
]

# %%
iss_topics.get_moving_average(df, window=3, rename_cols=True)[
    ["year", "no_of_projects", "no_of_projects_sma3"]
]

# %%
np.mean([10, 12, 8])

# %%
df[df.year != 2021].rolling(window=3, min_periods=1).mean()

# %%
(28 + 23 + 10) / 3

# %%
(28 + 9 + 8) / 3

# %%
df.set_index("year").loc[2016:2020, variable].mean()

# %%
iss_topics.get_moving_average(df, window=3, rename_cols=False)

# %%
get_growth_and_level_2(df[df.year.isin([2021]) == False], "no_of_projects")

# %%
cat = "Geothermal energy"
iss_figures.plot_bars(YEARLY_STATS, "no_of_projects", cat, "Number of projects")

# %%
# importlib.reload(iss_figures)

# %%
cat = "Hydrogen heating"
iss_figures.plot_bars(YEARLY_STATS, "no_of_projects", cat, "Number of projects")

# %%
cat = "Micro CHP"
iss_figures.plot_bars(YEARLY_STATS, "no_of_projects", cat, "Number of projects")

# %%
cat = "EEM"
iss_figures.plot_bars(YEARLY_STATS, "amount_total", cat, "Total amount")

# %%
cat = "Low carbon heating"
iss_figures.plot_bars(YEARLY_STATS, "amount_total", cat, "Total amount")

# %%
cat = "Heat pumps"

# %%
iss_figures.get_growth_and_level_std(YEARLY_STATS, cat=cat, variable="no_of_projects")

# %%
iss_figures.get_growth_and_level_std(YEARLY_STATS, cat=cat, variable="amount_total")

# %%
cat = "Low carbon heating"
iss_figures.get_growth_and_level_std(YEARLY_STATS, cat=cat, variable="amount_total")

# %%
iss_figures.get_growth_and_level_std(YEARLY_STATS, cat=cat, variable="amount_total")


# %% [markdown]
# ### Get project and funding benchmark

# %%
def get_growth_and_level_2(df, variable, year_1=2016, year_2=2020, window=3):
    df = df[df.year <= 2020]
    df_ma = iss_topics.get_moving_average(df, window=window, rename_cols=False)
    df = df.set_index("year")
    df_ma = df_ma.set_index("year")
    growth_rate = df_ma.loc[year_2, variable] / df_ma.loc[year_1, variable]
    level = df.loc[year_1:year_2, variable].mean()
    return growth_rate, level


# %%
def get_ma_values(df, variable, year_1=2016, year_2=2020, window=3):
    df = df[df.year <= 2020]
    df_ma = iss_topics.get_moving_average(df, window=window, rename_cols=False)
    df = df.set_index("year")
    df_ma = df_ma.set_index("year")
    return df_ma.loc[year_2, variable], df_ma.loc[year_1, variable]


# %%
def get_percent_increase(x):
    return (x[0] - x[1]) / x[1] * 100


# %%
gtr_funding_amounts = pd.read_csv(
    PROJECT_DIR / "outputs/GTR_funds.csv", names=["i", "doc_id", "amount"]
)

# %%
gtr_projects_ = gtr_projects.merge(
    gtr_funding_amounts, left_on="project_id", right_on="doc_id", how="left"
)

# %%
# gtr_projects_

# %%
gtr_projects_["year"] = gtr_projects_.start.apply(iss.convert_date_to_year)
gtr_total_projects = gtr_projects_.groupby("year").count().reset_index()
gtr_total_projects_amounts = gtr_projects_.groupby("year").sum().reset_index()

# %%
get_growth_and_level_2(gtr_total_projects, "project_id", year_1=2016, year_2=2020)

# %%
get_growth_and_level_2(gtr_total_projects_amounts, "amount", year_1=2016, year_2=2020)

# %%
get_ma_values(gtr_total_projects_amounts, "amount", year_1=2016, year_2=2020)

# %%
(4476675022.333333 - 3457875821.0) / 3457875821.0

# %%
plt.plot(gtr_total_projects_amounts.year, gtr_total_projects_amounts.amount)

# %%
df = gtr_total_projects_amounts
x = (df[df.year == 2020].amount.iloc[0], df[df.year == 2016].amount.iloc[0])
(x[0] - x[1]) / x[1]

# %%
x = get_ma_values(df, "amount", year_1=2016, year_2=2020, window=3)
(x[0] - x[1]) / x[1]

# %%
get_ma_values(
    YEARLY_STATS["Heat pumps"], "amount_total", year_1=2016, year_2=2020, window=3
)

# %%
(19442 - 2289.9863333333337) / 2289.9863333333337

# %%
# df=YEARLY_STATS['Heat pumps']
# df[df.year==2020].amount_total.iloc[0] / df[df.year==2017].amount_total.iloc[0]

# %%
cat = "Low carbon heating"
x = get_ma_values(YEARLY_STATS[cat], "amount_total", year_1=2016, year_2=2020, window=3)

# %%
iss_figures.get_growth_and_level_std(YEARLY_STATS, cat=cat, variable="amount_total")

# %%
(x[0] - x[1]) / x[1]

# %% [markdown]
# ### Get crunchbase benchmark

# %%
len(cb_df)

# %%
cb_df_rounds = iss.get_cb_org_funding_rounds(cb_df, cb_funding_rounds)

# %%
cb_df_yearly_funding = iss.get_cb_funding_per_year(cb_df_rounds)

# %%
cb_df_yearly_funding.head(2)

# %%
get_growth_and_level_2(cb_df_yearly_funding, variable="raised_amount_gbp_total")

# %%
get_growth_and_level_2(cb_df_yearly_funding, variable="no_of_rounds")

# %%
get_percent_increase(
    get_ma_values(cb_df_yearly_funding, variable="raised_amount_gbp_total")
)

# %%
df = cb_df_yearly_funding
get_percent_increase(
    (
        df[df.year == 2020].raised_amount_gbp_total.iloc[0],
        df[df.year == 2016].raised_amount_gbp_total.iloc[0],
    )
)

# %% [markdown]
# ## Matrix

# %%
importlib.reload(iss_figures)

# %%
variable = "no_of_projects"
y_label = "Growth"
x_label = f"Avg number of {variable} per year"
x_label = "Average number of projects per year"
cats = sorted(
    [
        "Heat pumps",
        "Biomass heating",
        "Hydrogen heating",
        "Geothermal energy",
        "Solar thermal",
        "District heating",
        "Heat storage",
        "Insulation & retrofit",
        "Energy management",
        "Micro CHP",
    ]
)
fig = iss.nicer_axis(
    iss_figures.plot_matrix(YEARLY_STATS, variable, cats, x_label, y_label)
)
fig

# %%
df_stats = pd.DataFrame(
    [
        iss_figures.get_growth_and_level_std(
            YEARLY_STATS, cat, variable, year_1=2016, year_2=2020, window=3
        )
        + tuple([cat])
        for cat in cats
    ],
    columns=["growth", variable, "std_dev", "tech_category"],
)

# %%
print(variable)
df_stats_all = get_year_by_year_stats(YEARLY_STATS, variable, year_dif=4)

# %%
df_stats.head(3)

# %%
df_stats

# %%
df_stats["tech_label"] = df_stats["tech_category"].copy()
df_stats.loc[
    df_stats.tech_category == "Biomass heating", "tech_label"
] = "Biomass\nheating"
df_stats.loc[
    df_stats.tech_category == "Hydrogen heating", "tech_label"
] = "Hydrogen\nheating"
df_stats.loc[
    df_stats.tech_category == "Geothermal energy", "tech_label"
] = "Geothermal\nenergy"
df_stats.loc[
    df_stats.tech_category == "Insulation & retrofit", "tech_label"
] = "Insulation\n& retrofit"
df_stats.loc[
    df_stats.tech_category == "Energy management", "tech_label"
] = "Energy\nmanagement"
# df_stats.loc[df_stats.tech_category=='Heat pumps', 'tech_label'] = 'Heat\npumps'
# df_stats.loc[df_stats.tech_category=='Heat storage', 'tech_label'] = 'Heat\nstorage'

# %%
fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
# ax.figure(figsize=(6, 6), dpi=80)
plt.errorbar(
    df_stats[variable],
    df_stats.growth,
    fmt="o",
    markersize=6,
    c=dark_purple,
    #     xerr = df_stats.std_dev
)

for cat in ["Heat pumps", "District heating", "Insulation & retrofit"]:
    df = df_stats_all[df_stats_all.tech_category == cat]
    plt.plot(df[variable], df.growth, ".--")

plt.plot([0, 30], [1, 1], "--", c="k", linewidth=0.5)

plt.plot([0, 30], [1.53, 1.53], "--", c="k", linewidth=0.5)

plt.xlim(0, 30)
plt.ylim(0, 3.5)

for i, row in df_stats.iterrows():
    plt.annotate(
        row.tech_label,
        (row[variable], row.growth - 0.05),
        fontsize=11,
        va="top",
        ha="center",
    )

plt.xlabel("Average number of new projects per year", fontsize=fontsize_med)
plt.ylabel("Growth", fontsize=fontsize_med)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.savefig(
    PROJECT_DIR / "outputs/figures/blog_figures" / "matrix_LCH_EEM_projects_2.svg",
    format="svg",
)
plt.show()

# %%
df_stats = pd.DataFrame(
    [
        iss_figures.get_growth_and_level(
            YEARLY_STATS, cat, "amount_total", year_1=2016, year_2=2020, window=3
        )
        + tuple([cat])
        for cat in cats
    ],
    columns=["growth", variable, "tech_category"],
)
df_stats["tech_label"] = df_stats["tech_category"].copy()

fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
# ax.figure(figsize=(6, 6), dpi=80)
plt.errorbar(
    df_stats[variable],
    df_stats.growth,
    fmt="o",
    markersize=6,
    c=dark_purple,
    #     xerr = df_stats.std_dev
)

plt.plot([0, 16000], [1, 1], "--", c="k", linewidth=0.5)
plt.plot([0, 16000], [1.295, 1.295], "--", c="k", linewidth=0.5)

plt.xlim(0, 16000)
plt.ylim(0, 9)

for i, row in df_stats.iterrows():
    plt.annotate(
        row.tech_label,
        (row[variable], row.growth - 0.05),
        fontsize=11,
        va="top",
        ha="center",
    )

plt.xlabel("Average yearly research funding ()", fontsize=fontsize_med)
plt.ylabel("Growth", fontsize=fontsize_med)

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.savefig(
    PROJECT_DIR / "outputs/figures/blog_figures" / "matrix_LCH_EEM_funding.svg",
    format="svg",
)

plt.show()

# %%
# plt.figure(figsize=(6, 6), dpi=80)
# plt.scatter(
#     df_stats[variable],
#     df_stats.growth,
# )

# %%
df_stats

# %%
# iss.nicer_axis(iss_figures.plot_matrix(YEARLY_STATS, variable, cats, x_label, y_label, window=5))

# %%
variable = "no_of_projects"
df_stats_all = get_year_by_year_stats(YEARLY_STATS, variable, year_dif=4)
y_label = "Growth"
x_label = "Avg number of projects per year"
cats = sorted(
    [
        "Heat pumps",
        "Biomass heating",
        "Hydrogen heating",
        "Geothermal energy",
        "Solar thermal",
        "District heating",
        "Heat storage",
        "Building insulation",
        "Energy management",
    ]
)
cats = sorted(
    [
        "Heat pumps",
        "Hydrogen heating",
        "Biomass heating",
        "Insulation & retrofit",
        "Energy management",
        "District heating",
    ]
)
iss.nicer_axis(
    plot_matrix_trajectories(
        df_stats_all, variable, cats, x_label, y_label, ww=400, hh=400
    ).interactive()
)

# %%
# YEARLY_STATS['EEM'] = YEARLY_STATS['Energy efficiency & management']

# %%
variable = "amount_total"
y_label = "Growth"
x_label = "Average new funding per year (£1000s)"
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "LCH & EEM",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
]

# %%
df_stats = pd.DataFrame(
    [
        iss_figures.get_growth_and_level_std(
            YEARLY_STATS, cat, variable, year_1=2016, year_2=2020, window=3
        )
        + tuple([cat])
        for cat in cats
    ],
    columns=["growth", variable, "std_dev", "tech_category"],
)
df_stats["tech_label"] = df_stats["tech_category"].copy()

# %%
fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
# ax.figure(figsize=(6, 6), dpi=80)
plt.errorbar(
    df_stats[variable],
    df_stats.growth,
    fmt="o",
    markersize=6,
    c=dark_purple,
    #     xerr = df_stats.std_dev
)

# for cat in ['Heat pumps', 'District heating', 'Insulation & retrofit']:
#     df = df_stats_all[df_stats_all.tech_category==cat]
#     plt.plot(df[variable], df.growth, '.--')

plt.plot([0, 60000], [1, 1], "--", c="k", linewidth=0.5)
plt.plot([0, 60000], [1.295, 1.295], "--", c="k", linewidth=0.5)

plt.xlim(0, 60000)
plt.ylim(0, 3.75)

for i, row in df_stats.iterrows():
    plt.annotate(
        row.tech_label,
        (row[variable], row.growth - 0.05),
        fontsize=11,
        va="top",
        ha="center",
    )

plt.xlabel("Average new funding per year (£1000s)", fontsize=fontsize_med)
plt.ylabel("Growth", fontsize=fontsize_med)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.savefig(
    PROJECT_DIR / "outputs/figures/blog_figures" / "matrix_GREENTECH_funding.svg",
    format="svg",
)
plt.show()

# %%
variable = "amount_total"
variable = "no_of_projects"
y_label = "Growth"
x_label = f"Avg number of {variable} per year"
# x_label = "Average number of projects per year"

cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
]

fig = iss.nicer_axis(
    iss_figures.plot_matrix(
        YEARLY_STATS,
        variable,
        cats,
        x_label,
        y_label,
        color_pal=(PLOT_REF_CATEGORY_DOMAIN, PLOT_REF_CATEGORY_COLOURS),
    )
)
fig

# %%
df_viz = (
    df_all_yearly_stats[df_all_yearly_stats.year.isin(list(range(2016, 2021)))]
    .groupby("tech_category")
    .mean()
    .reset_index()
)
df_viz.amount_total = df_viz.amount_total / 1000

sort_order = [
    "Batteries",
    "Low carbon heating",
    "Wind & offshore",
    "Solar",
    "Bioenergy",
    "EEM",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
]

# %%
fig = (
    alt.Chart(
        df_viz[df_viz.tech_category.isin(cats)],
        width=200,
    )
    .mark_bar()
    .encode(
        y=alt.Y("tech_category", title="", sort=sort_order),
        x=alt.X(
            "no_of_projects",
            title="Number of new projects",
        ),
    )
)
fig = iss.nicer_axis(fig)
fig

# %%
alt_save.save_altair(fig, "blog_fig_Barplot_Research_no_projects", driver)

# %%

# %%
fig = (
    alt.Chart(
        df_viz[df_viz.tech_category.isin(cats)],
        width=200,
    )
    .mark_bar()
    .encode(
        y=alt.Y("tech_category", title="", sort=sort_order),
        x=alt.X(
            "amount_total",
            title="Funding amount",
            scale=alt.Scale(domain=[0, 60])
            #             stack="normalize",
            #             title="Total amount raised ($1000s)"
        ),
        #         color="tech_category",
    )
)
fig = iss.nicer_axis(fig)
fig

# %%
alt_save.save_altair(fig, "blog_fig_Barplot_Research_amount", driver)

# %%

# %% [markdown]
# # Business investment

# %%
CB_DOCS_ALL_.tech_category.unique()

# %%
df = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(["LCH & EEM"])]

# %%
df = CB_DOCS_ALL_[
    CB_DOCS_ALL_.tech_category.isin(
        [
            "Batteries",
            "Hydrogen & fuel cells",
            "Carbon capture & storage",
            "Bioenergy",
            "Solar",
            "Low carbon heating",
            "Wind & offshore",
            "Energy efficiency & management",
            "Heating (other)",
        ]
    )
]

# %%
# iss.get_cb_org_funding_rounds(df, cb_funding_rounds).sort_values('announed_on')

# %%
len(df)

# %%
(len(df) - df.total_funding.isnull().sum()) / len(df)

# %%
(len(cb_df) - cb_df.total_funding.isnull().sum()) / len(cb_df)

# %%
YEARLY_STATS["Heat pumps"].head(1)

# %% [markdown]
# ### Low carbon heating figures

# %%
cat = "Low carbon heating"

# %%
iss_figures.get_growth_and_level_std(
    YEARLY_STATS, cat=cat, variable="raised_amount_gbp_total"
)

# %%
iss_figures.get_growth_and_level_std(YEARLY_STATS, cat=cat, variable="no_of_rounds")

# %%
cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == cat]
dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)
# dff[-dff.raised_amount.isnull()].sort_values("raised_amount")

# %%
dff.head(1)

# %%
dff[
    (dff.announced_on_date >= "2016-01-01") & (dff.announced_on_date <= "2020-12-31")
].raised_amount_gbp.median()

# %%
dff[(dff.announced_on_date >= "2016-01-01") & (dff.announced_on_date <= "2021-12-31")]

# %%
lch_investment = dff[
    (dff.announced_on_date >= "2007-01-01") & (dff.announced_on_date <= "2021-12-31")
]
lch_investment = lch_investment[
    lch_investment.funding_round_id != "32c568e7-ed7c-4b6b-bd8a-d45703a21b03"
]
lch_investment_yearly = iss.get_cb_funding_per_year(lch_investment.copy())

# %%
df = lch_investment_yearly
iss_figures.get_growth_and_level_std_(df, variable="raised_amount_gbp_total")

# %% [markdown]
# ### EEM figures

# %%
cat = "EEM"

# %%
iss_figures.get_growth_and_level_std(YEARLY_STATS, cat="EEM", variable="no_of_rounds")

# %%
iss_figures.get_growth_and_level_std(
    YEARLY_STATS, cat="EEM", variable="raised_amount_gbp_total"
)

# %%
x = get_ma_values(YEARLY_STATS[cat], variable="raised_amount_gbp_total")
(x[0] - x[1]) / x[1]

# %%
cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == cat]
dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)
# dff[-dff.raised_amount.isnull()].sort_values("raised_amount")

# %%
dff.head(1)

# %%
dff[
    (dff.announced_on_date >= "2016-01-01") & (dff.announced_on_date <= "2020-12-31")
].raised_amount_gbp.median()

# %%
dff_ = dff[
    (dff.announced_on_date >= "2016-01-01") & (dff.announced_on_date <= "2020-12-31")
]
dff_[-dff_.raised_amount_gbp.isnull()].sort_values("raised_amount_gbp").tail(15)

# %%
dff[(dff.announced_on_date >= "2016-01-01") & (dff.name == "Q-Bot")]

# %% [markdown]
# ### Other categories

# %%
df = YEARLY_STATS["EEM"]
df[df.year == 2020].raised_amount_gbp_total.iloc[0] / df[
    df.year == 2016
].raised_amount_gbp_total.iloc[0]

# %%
df = YEARLY_STATS["Low carbon heating"]
df[df.year == 2020].raised_amount_gbp_total.iloc[0] / df[
    df.year == 2016
].raised_amount_gbp_total.iloc[0]

# %%
df = YEARLY_STATS["Hydrogen & fuel cells"]
cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == "Hydrogen & fuel cells"]
dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)
# dff[-dff.raised_amount.isnull()].sort_values("raised_amount")
# dff.tail(10)

# %% [markdown]
# ### Figures

# %%
# cats = ['Batteries', 'Hydrogen & Fuel Cells',
#         'Carbon Capture & Storage', 'Bioenergy',
#         'Solar', 'Heating (all)', 'Building Energy Efficiency', 'Wind & Offshore', 'Heating (other)']
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
    "Heating (other)",
]

fig = (
    alt.Chart(
        df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)], width=500
    )
    .mark_bar()
    .encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y(
            "sum(raised_amount_usd_total)",
            stack="normalize",
            title="Total amount raised ($1000s)",
        ),
        color="tech_category",
    )
)
iss.nicer_axis(fig)

# %%
importlib.reload(iss_figures)

# %%
fig = iss.nicer_axis(
    iss_figures.plot_matrix(
        YEARLY_STATS,
        "no_of_rounds",
        cats,
        "Average number of deals",
        "Growth index",
        color_pal=(PLOT_REF_CATEGORY_DOMAIN, PLOT_REF_CATEGORY_COLOURS),
    )
)
fig

# %%
fig = iss.nicer_axis(
    iss_figures.plot_matrix(
        YEARLY_STATS,
        "raised_amount_gbp_total",
        cats,
        "Average number of deals",
        "Growth index",
        color_pal=(PLOT_REF_CATEGORY_DOMAIN, PLOT_REF_CATEGORY_COLOURS),
    )
)
fig

# %%
df_stats = pd.DataFrame(
    [
        iss_figures.get_growth_and_level(
            YEARLY_STATS, cat, "no_of_rounds", year_1=2016, year_2=2020, window=3
        )
        + tuple([cat])
        for cat in cats
    ],
    columns=["growth", variable, "tech_category"],
)
df_stats["tech_label"] = df_stats["tech_category"].copy()

fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
# ax.figure(figsize=(6, 6), dpi=80)
plt.errorbar(
    df_stats[variable],
    df_stats.growth,
    fmt="o",
    markersize=6,
    c=dark_purple,
    #     xerr = df_stats.std_dev
)
#
plt.plot([0, 20], [1, 1], "--", c="k", linewidth=0.5)
plt.plot([0, 20], [1.187, 1.187], "--", c="k", linewidth=0.5)

plt.xlim(0, 20)
plt.ylim(0, 3)

for i, row in df_stats.iterrows():
    plt.annotate(
        row.tech_label,
        (row[variable], row.growth - 0.05),
        fontsize=11,
        va="top",
        ha="center",
    )

plt.xlabel("Average number of deals", fontsize=fontsize_med)
plt.ylabel("Growth index", fontsize=fontsize_med)

plt.xticks([0, 5, 10, 15, 20], fontsize=11)
plt.yticks(fontsize=11)

plt.savefig(
    PROJECT_DIR / "outputs/figures/blog_figures" / "matrix_Investment.svg", format="svg"
)

plt.show()

# %%
# df_viz = df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)]
df_viz = df_all_yearly_stats_norm[df_all_yearly_stats_norm.tech_category.isin(cats)]
fig = (
    alt.Chart(df_viz, width=450, height=250)
    .mark_line(opacity=1)
    .encode(
        x=alt.X(
            "year:O", title="Year"
        ),  # scale=alt.Scale(domain=list(range(2015, 2021)))),
        y=alt.Y(
            #             "raised_amount_gbp_total",
            "no_of_rounds",
            #                         scale=alt.Scale(type='symlog'),
            #                         stack="normalize",
            title="Number of investment deals",
            #             scale=alt.Scale(domain=(0, 60000)),
        ),
        color="tech_category",
    )
)
fig = iss.nicer_axis(fig)
fig

# %%
alt_save.save_altair(fig, "blog_fig_Lineplot_Investment_deals", driver)

# %%
importlib.reload(iss)

# %%
# df_viz = df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)]
df_viz = df_all_yearly_stats_norm[df_all_yearly_stats_norm.tech_category.isin(cats)]
fig = (
    alt.Chart(df_viz, height=250, width=450)
    .mark_line(opacity=1)
    .encode(
        x=alt.X(
            "year:O", title="Year"
        ),  # scale=alt.Scale(domain=list(range(2015, 2021)))),
        y=alt.Y(
            "raised_amount_gbp_total",
            #             "no_of_rounds",
            #             "amount_total"
            #                         scale=alt.Scale(type='symlog'),
            #                         stack="normalize",
            title="Number of investment deals",
            scale=alt.Scale(domain=(0, 300000)),
        ),
        color="tech_category",
    )
)
fig = iss.nicer_axis(fig)
fig

# %%
alt_save.save_altair(fig, "blog_fig_Lineplot_Investment_amount", driver)

# %%
6 / 7

# %%
df_viz

# %%
CB_DOCS_ALL_.info()

# %%
CB_DOCS_ALL_[CB_DOCS_ALL_.name == "Mytrah Energy"].region


# %%
cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == "Wind & offshore"]
dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)
dff[-dff.raised_amount.isnull()].sort_values("raised_amount")

# %%

# %%

# %%
# CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(cats)].doc_id.duplicated().sum()

# %%
cats = sorted(
    [
        "Heat pumps",
        "Biomass heating",
        "Hydrogen heating",
        "Geothermal energy",
        "Solar thermal",
        "District heating",
        "Heat storage",
        "Building insulation",
        "Energy management",
        "Micro CHP",
    ]
)

# %%
x = []
for cat in cats:
    df = YEARLY_STATS[cat]
    df = df[df.year.isin(list(range(2016, 2021)))]
    df_sum = df.mean()
    x += [(cat, df_sum.amount_total, df_sum.no_of_projects, "research")] + [
        (cat, df_sum.raised_amount_usd_total, df_sum.no_of_rounds, "investment")
    ]

# %%
df = pd.DataFrame(
    x, columns=["tech_category", "amount", "no_of_projects_deals", "amount_category"]
)

# %%
df[df.amount_category == "research"]

# %%
df[df.amount_category == "research"].amount.median()

# %%
df[df.amount_category == "research"].no_of_projects_deals.median()

# %%
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
    "Heating (other)",
]

# %%
x = []
for cat in cats:
    df = YEARLY_STATS[cat]
    df = df[df.year.isin(list(range(2016, 2021)))]
    df_sum = df.mean()
    x += [(cat, df_sum.amount_total, df_sum.no_of_projects, "research")] + [
        (cat, df_sum.raised_amount_usd_total, df_sum.no_of_rounds, "investment")
    ]
df = pd.DataFrame(
    x, columns=["tech_category", "amount", "no_of_projects_deals", "amount_category"]
)
df[df.amount_category == "research"].amount.median()

# %%
df

# %%
df[df.amount_category == "investment"].no_of_projects_deals.median()

# %%
x = []
for cat in cats:
    df = YEARLY_STATS[cat]
    df = df[df.year.isin(list(range(2016, 2021)))]
    df_sum = df.sum()
    x += [(cat, df_sum.amount_total, df_sum.no_of_projects, "research")] + [
        (cat, df_sum.raised_amount_usd_total, df_sum.no_of_rounds, "investment")
    ]

# %%
df = pd.DataFrame(
    x, columns=["tech_category", "amount", "no_of_projects_deals", "amount_category"]
)

# %%
df_research = df[df.amount_category == "research"].copy()
df_investment = df[df.amount_category == "investment"].copy()

# %%
df_research.amount.sum()

# %%
df_gtr = GTR_DOCS_ALL_[
    (GTR_DOCS_ALL_.start >= "2016-01-01") & (GTR_DOCS_ALL_.start <= "2020-12-31")
]
df_gtr = df_gtr[df_gtr.tech_category.isin(cats)]
df_gtr.amount.sum() / 1000

# %%
df_cb = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(cats)]

# %%
# df_cb[df_cb.doc_id.duplicated(keep=False)].sort_values(['title'])

# %%
dff = iss.get_cb_org_funding_rounds(df_cb.copy(), cb_funding_rounds)
dff = dff[
    (dff.announced_on_date >= "2016-01-01") & (dff.announced_on_date <= "2020-12-31")
]
dff.raised_amount_gbp.sum()


# %%
def get_rounds(df):
    dff = iss.get_cb_org_funding_rounds(df, cb_funding_rounds)
    dff = dff[
        (dff.announced_on_date >= "2016-01-01")
        & (dff.announced_on_date <= "2020-12-31")
    ]
    return dff


# %%
# df_test = df_cb[df_cb.tech_category.isin(['Solar'])]
# df_test_rounds = get_rounds(df_test)

# %%
df_research["amount_proportion"] = df_research.amount / (df_gtr.amount.sum() / 1000)
df_investment["amount_proportion"] = df_investment.amount / dff.raised_amount_gbp.sum()

# %%
# df_investment.amount.sum()

# %%
# df_research_investment = pd.concat([df_research, df_investment])

# %%
fig = (
    alt.Chart(df, width=200)
    .mark_bar()
    .encode(
        x=alt.X("amount_category", title=""),
        y=alt.Y("sum(amount)", stack="normalize", title="Fraction of money"),
        color="tech_category",
    )
)
fig = iss.nicer_axis(fig)
fig

# %%
df_research_investment = df_research.merge(
    df_investment[["tech_category", "amount_proportion"]], on="tech_category"
)
df_research_investment = df_research_investment.rename(
    columns={"amount_proportion_x": "research", "amount_proportion_y": "investment"}
)
df_research_investment = df_research_investment[
    df_research_investment.tech_category != "Heating (other)"
]
df_research_investment = df_research_investment.set_index("tech_category")
df_research_investment = df_research_investment[["research", "investment"]]
df_research_investment = df_research_investment.sort_values("research")[
    ["investment", "research"]
]

# %%
df_research_investment

# %%
# fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
fig = df_research_investment.plot.barh(figsize=(6, 7), width=0.75)
plt.xlim(0, 0.6)
plt.grid(axis="x")
plt.savefig(
    PROJECT_DIR / "outputs/figures/blog_figures" / "barplots_proportion_funding.svg",
    format="svg",
)

# %%
cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
    "Heating (other)",
]

# %%
x = []
for cat in cats:
    df = YEARLY_STATS[cat]
    df = df[df.year.isin(list(range(2016, 2021)))]
    df_sum = df.mean()
    x += [(cat, df_sum.amount_total, df_sum.no_of_projects, "research")] + [
        (cat, df_sum.raised_amount_usd_total, df_sum.no_of_rounds, "investment")
    ]
df = pd.DataFrame(
    x, columns=["tech_category", "amount", "no_of_projects_deals", "amount_category"]
)
df[df.amount_category == "research"].amount.median()

# %%

# %%

# %%
# df_viz = df_research_investment[
#     df_research_investment.tech_category != "Heating (other)"
# ]
# alt.Chart(df_research_investment).mark_bar().encode(
#     x=alt.X("amount_proportion"),
#     y="amount_category",
#     color="amount_category",
#     row="tech_category",
# )

# %%
# # set width of bars
# barWidth = 0.25

# # set heights of bars
# bars1 = [12, 30, 1, 8, 22]
# bars2 = [28, 6, 16, 5, 10]
# bars3 = [29, 3, 24, 25, 17]

# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]

# # Make the plot
# plt.bar(r, color='#7f6d5f', width=barWidth, edgecolor='white', label='var1')
# plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='var2')
# plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')

# # Add xticks on the middle of the group bars
# plt.xlabel('group', fontweight='bold')
# plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])

# # Create legend & Show graphic
# plt.legend()
# plt.show()

# %%
# fig = (
#     alt.Chart(df_research_investment, width=200)
#     .mark_bar()
#     .encode(
#         x=alt.X("amount_category", title=""),
#         y=alt.Y("sum(amount)", stack="normalize", title="Fraction of money"),
#         color="tech_category",
#     )
# )
# fig = iss.nicer_axis(fig)
# fig

# %%
cats = [
    "Heat pumps",
    "Biomass heating",
    "Hydrogen heating",
    "Geothermal energy",
    "Solar thermal",
    "District heating",
    "Heat storage",
    "Insulation & retrofit",
    "Energy management",
    "Micro CHP",
    "Heating (other)",
]

x = []
for cat in cats:
    df = YEARLY_STATS[cat]
    df = df[df.year.isin(list(range(2016, 2021)))]
    df_sum = df.sum()
    x += [(cat, df_sum.amount_total, df_sum.no_of_projects, "research")] + [
        (cat, df_sum.raised_amount_usd_total, df_sum.no_of_rounds, "investment")
    ]
df = pd.DataFrame(
    x, columns=["tech_category", "amount", "no_of_projects_deals", "amount_category"]
)

# %%
df_investment = df[df.amount_category == "investment"][
    ["tech_category", "no_of_projects_deals", "amount_category"]
]
df_investment.no_of_projects_deals = df_investment.no_of_projects_deals.astype(int)
df_investment = df_investment.sort_values("no_of_projects_deals")

# %%
fig, ax = plt.subplots(figsize=(4, 6), dpi=80)
plt.barh(y=df_investment["tech_category"], width=df_investment["no_of_projects_deals"])
plt.grid(axis="x")
plt.xlabel("Number of investment deals")

# %%
# # fig, ax = plt.subplots(figsize=(6, 6), dpi=80)
# fig = df_investment.plot.barh(figsize=(6, 7), y='tech_category', width='no_of_projects_deals')
# # plt.xlim(0, 80)
# # plt.grid(axis="x")
# # plt.savefig(
# #     PROJECT_DIR / "outputs/figures/blog_figures" / "barplots_proportion_funding.svg",
# #     format="svg",
# # )

# %%
df_investment

# %%
alt_save.save_altair(fig, "blog_fig_Barplot_Research_vs_Investment", driver)

# %%
variable = "raised_amount_usd_total"
y_label = "Growth"
x_label = f"Avg number of {variable} per year"
# x_label = "Average number of projects per year"

cats = [
    "Batteries",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
    "Bioenergy",
    "Solar",
    "Low carbon heating",
    "Wind & offshore",
    "EEM",
]

fig = iss.nicer_axis(
    iss_figures.plot_matrix(
        YEARLY_STATS,
        variable,
        cats,
        x_label,
        y_label,
        color_pal=(PLOT_REF_CATEGORY_DOMAIN, PLOT_REF_CATEGORY_COLOURS),
    )
)
fig

# %%
variable = "raised_amount_usd_total"
# variable='no_of_rounds'
y_label = "Growth"
x_label = f"Avg number of {variable} per year"
# x_label = "Average number of projects per year"

cats = [
    "Heat pumps",
    "Biomass heating",
    "Hydrogen heating",
    "Geothermal energy",
    "Solar thermal",
    "District heating",
    "Heat storage",
    "Insulation & retrofit",
    "Energy management",
    "Micro CHP",
]

fig = iss.nicer_axis(
    iss_figures.plot_matrix(
        YEARLY_STATS,
        variable,
        cats,
        x_label,
        y_label,
    )
)
fig.interactive()

# %%
cat = "Low carbon heating"
iss_figures.plot_bars(
    YEARLY_STATS, "raised_amount_usd_total", cat, "Number of projects"
)

# %%
cb_selected_cat = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category == "Low carbon heating"]
dff = iss.get_cb_org_funding_rounds(cb_selected_cat, cb_funding_rounds)
dff[-dff.raised_amount.isnull()].sort_values("announced_on")

# %% [markdown]
# ## Public discourse

# %%
discourse_dir = "/Users/karliskanders/Documents/innovation_sweet_spots/outputs/data/intermediate_ISS_outputs/"

# %% [markdown]
# ### Heat pumps

# %%
mentions_hp_guardian = pd.read_csv(
    discourse_dir + "term_mentions_guardian/mentions_df_hp.csv"
)
mentions_hp_hansard = pd.read_excel(
    discourse_dir + "term_mentions_hansard/mentions_df_hp_hansard.xlsx"
)

# %%
mentions_hp_hansard.head(2)

# %%
mentions_hp_guardian = mentions_hp_guardian.rename(columns={"Unnamed: 0": "year"})
mentions_hp_hansard = mentions_hp_hansard.rename(columns={"Unnamed: 0": "year"})

# %%
iss_topics.get_moving_average

# %%
mentions_hp_guardian = iss_topics.get_moving_average(
    mentions_hp_guardian, window=1, rename_cols=False
)
mentions_hp_hansard = iss_topics.get_moving_average(
    mentions_hp_hansard, window=1, rename_cols=False
)

# %%
fontsize_med = 12

# %%
fig, ax = plt.subplots(figsize=(6, 4), dpi=80)

plt.plot(
    mentions_hp_guardian.year,
    mentions_hp_guardian.total_documents
    #     fmt='o',
    #     markersize=6,
    #     c=dark_purple,
    #     xerr = df_stats.std_dev
)

# plt.plot(
#     mentions_hp_hansard.year,
#     mentions_hp_hansard.total_documents
# #     fmt='o',
# #     markersize=6,
# #     c=dark_purple,
# #     xerr = df_stats.std_dev
# )

# plt.plot([0, 30], [1, 1], '--', c='k', linewidth=0.5)

plt.xlim(2007, 2021)
plt.ylim(5, 35)

# for i, row in df_stats.iterrows():
#     plt.annotate(row.tech_label, (row[variable], row.growth-0.05), fontsize=11, va='top', ha='center')

plt.ylabel("Number of articles", fontsize=fontsize_med)
plt.xlabel("Year", fontsize=fontsize_med)

plt.xticks([2007, 2009, 2011, 2013, 2015, 2017, 2019, 2021], fontsize=11)
# plt.yticks(fontsize=11)

# plt.savefig(
#     PROJECT_DIR / "outputs/figures/blog_figures" / "discourse_heatpumps.svg",
#     format="svg",
# )

plt.show()

# %% [markdown]
# ### Hydrogen heating

# %%
df = pd.read_excel(discourse_dir + "term_mentions_guardian/mentions_comparison.xlsx")
df = df.rename(columns={"Unnamed: 0": "year"})

# %%
df.info()

# %%
fig, ax = plt.subplots(figsize=(6, 4), dpi=80)

plt.plot(df.year, df.prop_hydrogen * 100)

plt.plot(df.year, df.prop_hydrogen_heating * 100)

plt.plot(df.year, df.prop_hp * 100)

# plt.plot(
#     mentions_hp_hansard.year,
#     mentions_hp_hansard.total_documents
# #     fmt='o',
# #     markersize=6,
# #     c=dark_purple,
# #     xerr = df_stats.std_dev
# )

# plt.plot([0, 30], [1, 1], '--', c='k', linewidth=0.5)

# plt.xlim(2007, 2021)
plt.ylim(0, 7)

# for i, row in df_stats.iterrows():
#     plt.annotate(row.tech_label, (row[variable], row.growth-0.05), fontsize=11, va='top', ha='center')

plt.ylabel("Percentage of articles", fontsize=fontsize_med)
plt.xlabel("Year", fontsize=fontsize_med)

plt.xticks(list(range(2007, 2022, 2)), fontsize=11)
# plt.yticks(fontsize=11)

plt.savefig(
    PROJECT_DIR / "outputs/figures/blog_figures" / "discourse_hydrogen_proportion.svg",
    format="svg",
)

plt.show()

# %%
plt.plot(df.year, df.total_environment)

# %%
year_1 = 2016
year_2 = 2020
var = "prop_hydrogen"
var = "total_hydrogen"
df[df.year == year_2][var].iloc[0] / df[df.year == year_1][var].iloc[0]

# %%
x = get_ma_values(df, "total_hydrogen", year_1=2016, year_2=2020, window=3)
(x[0] - x[1]) / x[1]

# %%
x = get_ma_values(df, "prop_hydrogen", year_1=2016, year_2=2020, window=3)
(x[0] - x[1]) / x[1]

# %%
# year_1=2008
# year_2=2021
# var = 'prop_hydrogen'
# df[df.year==year_2][var].iloc[0] /  df[df.year==year_1][var].iloc[0]

# %%
year_1 = 2017
year_2 = 2020
var = "prop_hydrogen_heating"
# var = 'total_hydrogen_heating'
df[df.year == year_2][var].iloc[0] / df[df.year == year_1][var].iloc[0]

# %%
x = get_ma_values(df, "prop_hydrogen_heating", year_1=2016, year_2=2020, window=3)
(x[0] - x[1]) / x[1]

# %%
year_1 = 2016
year_2 = 2020
var = "prop_hp"
df[df.year == year_2][var].iloc[0] / df[df.year == year_1][var].iloc[0]

# %%
year_1 = 2016
year_2 = 2020
var = "total_hp"
df[df.year == year_2][var].iloc[0] / df[df.year == year_1][var].iloc[0]

# %%
year_1 = 2016
year_2 = 2020
var = "prop_hp"
df[df.year == year_2][var].iloc[0] / df[df.year == year_1][var].iloc[0]

# %%
x = get_ma_values(df, "prop_hp", year_1=2016, year_2=2020, window=3)
print(x[0] / x[1])
(x[0] - x[1]) / x[1] * 100

# %%
year_1 = 2008
year_2 = 2016
var = "prop_hp"
df[df.year == year_2][var].iloc[0] / df[df.year == year_1][var].iloc[0]

# %%
year_1 = 2008
year_2 = 2016
var = "total_hp"
df[df.year == year_2][var].iloc[0] / df[df.year == year_1][var].iloc[0]

# %%
x = get_ma_values(df, "total_hp", year_1=2008, year_2=2016, window=3)
(x[0] - x[1]) / x[1] * 100

# %%
x = get_ma_values(df, "prop_hp", year_1=2008, year_2=2016, window=3)
(x[0] - x[1]) / x[1] * 100

# %%
x = get_ma_values(df, "total_hp", year_1=2008, year_2=2016, window=1)
(x[0] - x[1]) / x[1] * 100

# %%
x = get_ma_values(df, "total_hp", year_1=2016, year_2=2020, window=1)
(x[0] - x[1]) / x[1] * 100

# %%
# mentions_hp_guardian = pd.read_csv(
#     discourse_dir + "term_mentions_guardian/mentions_df_hp.csv"
# )
# mentions_hp_hansard = pd.read_excel(
#     discourse_dir + "term_mentions_hansard/mentions_df_hp_hansard.xlsx"
# )

# %% [markdown]
# ### Hansard

# %%
df = pd.read_excel(
    discourse_dir + "term_mentions_hansard/mentions_df_comparison_hansard(1).xlsx"
)
df = df.rename(columns={"Unnamed: 0": "year"})

# %%
df.info()

# %%
fig, ax = plt.subplots(figsize=(6, 4), dpi=80)

plt.plot(df.year, df.total_hydrogen)

plt.plot(df.year, df.total_hydrogen_heating)

plt.plot(df.year, df.total_hp)

# plt.plot(
#     mentions_hp_hansard.year,
#     mentions_hp_hansard.total_documents
# #     fmt='o',
# #     markersize=6,
# #     c=dark_purple,
# #     xerr = df_stats.std_dev
# )

# plt.plot([0, 30], [1, 1], '--', c='k', linewidth=0.5)

plt.xlim(2007, 2021)
# plt.ylim(0, 7)

# for i, row in df_stats.iterrows():
#     plt.annotate(row.tech_label, (row[variable], row.growth-0.05), fontsize=11, va='top', ha='center')

plt.ylabel("Number of speeches", fontsize=fontsize_med)
plt.xlabel("Year", fontsize=fontsize_med)

plt.xticks(list(range(2007, 2022, 2)), fontsize=11)
plt.legend(["Hydrogen", "Hydrogen heating", "Heat pumps"])
# plt.yticks(fontsize=11)
plt.ticklabel_format(useOffset=False, style="plain")

plt.savefig(
    PROJECT_DIR / "outputs/figures/blog_figures" / "discourse_hansard.svg",
    format="svg",
)

plt.show()

# %% [markdown]
# ### Small multiples

# %%
YEARLY_STATS["Heat pumps"]

# %%
cats = [
    "Heat pumps",
    "Hydrogen heating",
    "District heating",
    "Biomass heating",
    "Geothermal energy",
    "Solar thermal",
    #     "Heat storage",
    "Micro CHP",
    "Insulation & retrofit",
    "Energy management",
]

# %%
YEARLY_STATS["Solar thermal"]

# %%
iss_topics.get_moving_average(YEARLY_STATS[cat], window=3)

# %%

# %%
fig, ax = plt.subplots(figsize=(9, 6), dpi=80)
fig.tight_layout()

# Make a data frame
df = pd.DataFrame(
    {
        "x": range(1, 11),
        "y1": np.random.randn(10),
        "y2": np.random.randn(10) + range(1, 11),
        "y3": np.random.randn(10) + range(11, 21),
        "y4": np.random.randn(10) + range(6, 16),
        "y5": np.random.randn(10) + range(4, 14) + (0, 0, 0, 0, 0, 0, 0, -3, -8, -6),
        "y6": np.random.randn(10) + range(2, 12),
        "y7": np.random.randn(10) + range(5, 15),
        "y8": np.random.randn(10) + range(4, 14),
        "y9": np.random.randn(10) + range(4, 14),
    }
)

# Initialize the figure style
# plt.style.use('seaborn-whitegrid')
plt.style.use("classic")
plt.rcParams["svg.fonttype"] = "none"

# create a color palette
palette = plt.get_cmap("Set1")

# multiple line plot
num = 0
# for column in df.drop('x', axis=1):
for cat in cats:
    num += 1

    # Find the right spot on the plot
    plt.subplot(3, 3, num)

    yearly_df = YEARLY_STATS[cat].copy().fillna(0)
    y = iss_topics.get_moving_average(yearly_df, window=3, rename_cols=False).articles
    y.fillna(0)
    y = y / np.max(y)

    y2 = iss_topics.get_moving_average(yearly_df, window=3, rename_cols=False).speeches
    y2 = y2 / np.max(y2)

    # Plot the lineplot
    plt.plot(
        yearly_df.year,
        y,
        marker="",
        color=palette(num),
        linewidth=1.9,
        alpha=0.9,
        label=column,
    )

    plt.plot(
        yearly_df.year,
        y2,
        marker="",
        ls="--",
        color=palette(num),
        linewidth=1.9,
        alpha=0.9,
        label=column,
    )

    # Same limits for every chart
    plt.xlim(2007, 2021)
    plt.ylim(0, 1.05)

    plt.xticks(range(2008, 2021, 4))
    plt.ticklabel_format(useOffset=False, style="plain")

    # Not ticks everywhere
    if num in range(7):
        plt.tick_params(labelbottom="off")
    if num not in [1, 4, 7]:
        plt.tick_params(labelleft="off")

    #     ax.spines['top'].set_visible(False)
    #     plt.spines['right'].set_visible(False)
    #     ax.spines['bottom'].set_visible(False)
    #     ax.spines['left'].set_visible(False)

    # Add title
    plt.title(cat, loc="left", fontsize=12, fontweight=0, color=palette(num))

# general title
# plt.suptitle("How the 9 students improved\nthese past few days?", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)

# Axis titles
# plt.text(0.5, 0.02, 'Time', ha='center', va='center')
# plt.text(0.06, 0.5, 'Note', ha='center', va='center', rotation='vertical')

plt.savefig(
    PROJECT_DIR / "outputs/figures/blog_figures" / "discourse_small_multiples.svg",
    format="svg",
)

# Show the graph
plt.show()

# %%
fig, ax = plt.subplots(figsize=(9, 6), dpi=80)
fig.tight_layout()

# Make a data frame
df = pd.DataFrame(
    {
        "x": range(1, 11),
        "y1": np.random.randn(10),
        "y2": np.random.randn(10) + range(1, 11),
        "y3": np.random.randn(10) + range(11, 21),
        "y4": np.random.randn(10) + range(6, 16),
        "y5": np.random.randn(10) + range(4, 14) + (0, 0, 0, 0, 0, 0, 0, -3, -8, -6),
        "y6": np.random.randn(10) + range(2, 12),
        "y7": np.random.randn(10) + range(5, 15),
        "y8": np.random.randn(10) + range(4, 14),
        "y9": np.random.randn(10) + range(4, 14),
    }
)

# Initialize the figure style
# plt.style.use('seaborn-whitegrid')
plt.style.use("classic")
plt.rcParams["svg.fonttype"] = "none"

# create a color palette
palette = plt.get_cmap("Set1")

# multiple line plot
num = 0
# for column in df.drop('x', axis=1):
for cat in cats:
    num += 1

    # Find the right spot on the plot
    plt.subplot(3, 3, num)

    yearly_df = YEARLY_STATS[cat].copy().fillna(0)
    y = iss_topics.get_moving_average(yearly_df, window=3, rename_cols=False).articles
    y.fillna(0)
    #     y = y / np.max(y)

    y2 = iss_topics.get_moving_average(yearly_df, window=3, rename_cols=False).speeches
    #     y2 = y2 / np.max(y2)

    # Plot the lineplot
    plt.plot(
        yearly_df.year,
        y,
        marker="",
        color=palette(num),
        linewidth=1.9,
        alpha=0.9,
        label=column,
    )

    plt.plot(
        yearly_df.year,
        y2,
        marker="",
        ls="--",
        color=palette(num),
        linewidth=1.9,
        alpha=0.9,
        label=column,
    )

    # Same limits for every chart
    plt.xlim(2007, 2021)
    #     plt.ylim(0, 1.05)

    plt.xticks(range(2008, 2021, 4))
    plt.ticklabel_format(useOffset=False, style="plain")

    # Not ticks everywhere
    if num in range(7):
        plt.tick_params(labelbottom="off")
    if num not in [1, 4, 7]:
        plt.tick_params(labelleft="off")

    #     ax.spines['top'].set_visible(False)
    #     plt.spines['right'].set_visible(False)
    #     ax.spines['bottom'].set_visible(False)
    #     ax.spines['left'].set_visible(False)

    # Add title
    plt.title(cat, loc="left", fontsize=12, fontweight=0, color=palette(num))

# general title
# plt.suptitle("How the 9 students improved\nthese past few days?", fontsize=13, fontweight=0, color='black', style='italic', y=1.02)

# Axis titles
# plt.text(0.5, 0.02, 'Time', ha='center', va='center')
# plt.text(0.06, 0.5, 'Note', ha='center', va='center', rotation='vertical')

# plt.savefig(
#     PROJECT_DIR / "outputs/figures/blog_figures" / "discourse_small_multiples.svg",
#     format="svg",
# )

# Show the graph
plt.show()

# %%
# df_viz = df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)]
df_viz = df_all_yearly_stats_norm[df_all_yearly_stats_norm.tech_category.isin(cats)]
fig = (
    alt.Chart(df_viz, width=450, height=250)
    .mark_line(opacity=1)
    .encode(
        x=alt.X(
            "year:O", title="Year"
        ),  # scale=alt.Scale(domain=list(range(2015, 2021)))),
        y=alt.Y(
            #             "raised_amount_gbp_total",
            "no_of_rounds",
            #                         scale=alt.Scale(type='symlog'),
            #                         stack="normalize",
            title="Number of investment deals",
            #             scale=alt.Scale(domain=(0, 60000)),
        ),
        color="tech_category",
    )
)
fig = iss.nicer_axis(fig)
fig

# %%
df_viz = (
    df_all_yearly_stats[df_all_yearly_stats.year.isin(list(range(2016, 2021)))]
    .groupby("tech_category")
    .mean()
    .reset_index()
)
df_viz.amount_total = df_viz.amount_total / 1000

sort_order = [
    "Batteries",
    "Low carbon heating",
    "Wind & offshore",
    "Solar",
    "Bioenergy",
    "EEM",
    "Hydrogen & fuel cells",
    "Carbon capture & storage",
]

fig = (
    alt.Chart(
        df_viz[df_viz.tech_category.isin(cats)],
        width=200,
    )
    .mark_bar()
    .encode(
        y=alt.Y("tech_category", title="", sort=sort_order),
        x=alt.X(
            "no_of_projects",
            title="Number of new projects",
        ),
    )
)
fig = iss.nicer_axis(fig)
fig

# %%
len(cb_df)

# %%
len(CB_DOCS_ALL_[-CB_DOCS_ALL_.total_funding.isnull()]) / len(CB_DOCS_ALL_)

# %%
len(cb_df[-cb_df.total_funding.isnull()]) / len(cb_df)

# %%
cb = crunchbase.get_crunchbase_orgs_full()
print(len(cb[-cb.total_funding.isnull()]) / len(cb))
del cb

# %%
