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

# %%
import innovation_sweet_spots.utils.io as iss_io

# %%
crunchbase.CB_PATH = crunchbase.CB_PATH.parent / "cb_2021"

# %%
# Import Crunchbase data
cb = crunchbase.get_crunchbase_orgs_full()
cb_df = cb[-cb.id.duplicated()]
cb_df = cb_df[cb_df.country == "United Kingdom"]
cb_df = cb_df.reset_index(drop=True)
del cb
cb_investors = crunchbase.get_crunchbase_investors()
cb_investments = crunchbase.get_crunchbase_investments()
cb_funding_rounds = crunchbase.get_crunchbase_funding_rounds()

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
    #     '#bab0ac',
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
def get_year_by_year_stats(YEARLY_STATS, variable, year_dif=4):
    df_stats_all = pd.DataFrame()
    for cat in list(YEARLY_STATS.keys()):
        for y in range(2011, 2021):
            yy = year_dif
            #         if y==2009: yy=1;
            #         if y==2010: yy=2;
            #         if y==2011: yy=2;
            #         if y>2010: yy=4;
            df_stats = pd.DataFrame(
                [
                    get_growth_and_level(cat, variable, y - yy, y) + tuple([cat, y])
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
from matplotlib import pyplot as plt
import seaborn as sns

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
# GTR_DOCS_ALL_[GTR_DOCS_ALL_.tech_category=='Hydrogen & fuel cells']

# %%
list(YEARLY_STATS.keys())

# %%
df_all_yearly_stats = pd.DataFrame()
for key in YEARLY_STATS:
    df_ = YEARLY_STATS[key].copy()
    df_["tech_category"] = key
    df_all_yearly_stats = df_all_yearly_stats.append(df_, ignore_index=True)

# df_all_yearly_stats_norm = pd.DataFrame()
# for key in YEARLY_STATS_NORM:
#     df_ = YEARLY_STATS_NORM[key].copy()
#     df_["tech_category"] = key
#     df_all_yearly_stats_norm = df_all_yearly_stats_norm.append(df_, ignore_index=True)

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

# %%
cat = "Insulation & retrofit"
iss_figures.plot_bars(YEARLY_STATS, "no_of_projects", cat, "Number of projects")

# %%
cat = "Geothermal energy"
iss_figures.plot_bars(YEARLY_STATS, "no_of_projects", cat, "Number of projects")

# %%
# importlib.reload(iss_figures)

# %%
cat = "Hydrogen heating"
iss_figures.plot_bars(YEARLY_STATS, "no_of_projects", cat, "Number of projects")

# %%

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
fontsize_med = 12
plt.rcParams["svg.fonttype"] = "none"

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

# plt.savefig(PROJECT_DIR / 'outputs/figures/blog_figures' / 'matrix_LCH_EEM_projects.svg', format='svg')
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

# plt.plot([0, 30], [1, 1], '--', c='k', linewidth=0.5)

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
plt.plot(dates, values, "-o")

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

# %%
fig = (
    alt.Chart(
        #         df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)],
        df_viz[df_viz.tech_category.isin(cats)],
        width=200,
    )
    .mark_bar()
    .encode(
        y=alt.X("tech_category", title=""),
        x=alt.Y(
            "no_of_projects",
            #             stack="normalize",
            #             title="Total amount raised ($1000s)"
        ),
        #         color="tech_category",
    )
)
fig = iss.nicer_axis(fig)
fig

# %%
fig = (
    alt.Chart(
        #         df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)],
        df_viz[df_viz.tech_category.isin(cats)],
        width=200,
    )
    .mark_bar()
    .encode(
        y=alt.X("tech_category", title=""),
        x=alt.Y(
            "amount_total",
            #             stack="normalize",
            #             title="Total amount raised ($1000s)"
        ),
        #         color="tech_category",
    )
)
iss.nicer_axis(fig)

# %%

# %% [markdown]
# # Business investment

# %%
CB_DOCS_ALL_.tech_category.unique()

# %%
df = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(["LCH & EEM"])]

# %%
df = CB_DOCS_ALL_[CB_DOCS_ALL_.tech_category.isin(["Low carbon heating"])]

# %%
df

# %%
len(df)

# %%
(len(df) - df.total_funding.isnull().sum()) / len(df)

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

# plt.plot([0, 30], [1, 1], '--', c='k', linewidth=0.5)

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

# %%
df_viz = df_all_yearly_stats[df_all_yearly_stats.tech_category.isin(cats)]
fig = (
    alt.Chart(df_viz, width=500)
    .mark_line(opacity=1)
    .encode(
        x=alt.X(
            "year:O", title="Year"
        ),  # scale=alt.Scale(domain=list(range(2015, 2021)))),
        y=alt.Y(
            "amount_total",
            #             scale=alt.Scale(type='symlog'),
            #             stack="normalize",
            title="Total amount raised ($1000s)",
            scale=alt.Scale(domain=(0, 60000)),
        ),
        color="tech_category",
    )
)
iss.nicer_axis(fig)

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
df

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
df[df.amount_category == "investment"].no_of_projects_deals.mean()

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

plt.savefig(
    PROJECT_DIR / "outputs/figures/blog_figures" / "discourse_heatpumps.svg",
    format="svg",
)

plt.show()

# %%
