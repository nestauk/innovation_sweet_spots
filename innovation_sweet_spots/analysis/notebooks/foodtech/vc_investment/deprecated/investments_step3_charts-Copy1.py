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
# # Food tech: Venture funding trends
# ## Step 3: Analysis and charts
#
# - Data has been fetched from Dealroom business intelligence database (step 1)
# - Companies and their category assignments have been reviewed and processed (step 2)
# - This notebook (step 3) produces charts for the report

# %% [markdown]
# ### Loading dependencies

# %%
import innovation_sweet_spots.analysis.wrangling_utils as wu
import innovation_sweet_spots.analysis.analysis_utils as au
from innovation_sweet_spots.utils import plotting_utils as pu
import innovation_sweet_spots.utils.text_cleaning_utils as tcu
from innovation_sweet_spots import PROJECT_DIR
from innovation_sweet_spots.utils.io import save_json, load_json

output_folder = PROJECT_DIR / "outputs/foodtech/venture_capital"

import altair as alt
import pandas as pd
import numpy as np
import utils
import importlib
from collections import defaultdict
import itertools

COLUMN_CATEGORIES = wu.dealroom.COLUMN_CATEGORIES

# %%
# Plotting utils
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver(path=alt_save.FIGURE_PATH + "/foodtech")

# Figure version name
VERSION_NAME = "October_VC"

# %%
# Initialise a Dealroom wrangler instance
DR = wu.DealroomWrangler()

# Check the number of companies
len(DR.company_data)

# %% [markdown]
# ### Import reviewed data

# %%
# Taxonomy file for the VC data
taxonomy_df = pd.read_csv(output_folder / "vc_tech_taxonomy.csv")
# Mapping from minor sub categories to major categories
minor_to_major = load_json(output_folder / "vc_tech_taxonomy_minor_to_major.json")
# Mapping from companies to taxonomy categories
company_to_taxonomy_df = pd.read_csv(output_folder / "vc_company_to_taxonomy.csv")
company_to_taxonomy_df.id = company_to_taxonomy_df.id.astype(str)

# %%
# # Uncomment if doing the analysis soley for the UK
# uk_ids = DR.company_data.query("country == 'United Kingdom'").id.to_list()
# company_to_taxonomy_df = company_to_taxonomy_df.query("id in @uk_ids")
# VERSION_NAME = "October_VC_UK"

# %%
len(company_to_taxonomy_df)

# %% [markdown]
# ### Check the different investment deal types

# %%
for d in sorted(utils.EARLY_DEAL_TYPES):
    print(d)

# %%
for d in sorted(utils.LATE_DEAL_TYPES):
    print(d)

# %%
importlib.reload(utils)

# %%
category_ids = get_category_ids(taxonomy_df, rejected_tags, DR, column="Minor")
# ids = category_ids['logistics']

# %%
early_vs_late_deals = pd.concat(
    [
        (
            get_category_ts(category_ids, DR, deal_type=utils.EARLY_DEAL_TYPES).assign(
                deal_type="early"
            )
        ),
        (
            get_category_ts(category_ids, DR, deal_type=utils.LATE_DEAL_TYPES).assign(
                deal_type="late"
            )
        ),
    ],
    ignore_index=True,
)

# %%
early_vs_late_deals_5y = (
    early_vs_late_deals.query("year >= 2017 and year < 2022")
    .groupby(["Category", "deal_type"], as_index=False)
    .agg(no_of_rounds=("no_of_rounds", "sum"))
)
total_deals = early_vs_late_deals_5y.groupby("Category", as_index=False).agg(
    total_no_of_rounds=("no_of_rounds", "sum")
)
early_vs_late_deals_5y = early_vs_late_deals_5y.merge(total_deals).assign(
    fraction=lambda df: df["no_of_rounds"] / df["total_no_of_rounds"]
)

# %%
alt.Chart((early_vs_late_deals_5y.query("deal_type=='late'")),).mark_bar().encode(
    # x=alt.X('no_of_rounds:Q'),
    x=alt.X("fraction:Q"),
    y=alt.Y("Category:N", sort="-x"),
    # color='deal_type',
    tooltip=["deal_type", "no_of_rounds"],
)

# %%

# %%
cat = "diet"
alt.Chart((early_vs_late_deals.query("Category==@cat")),).mark_bar().encode(
    x=alt.X("year:O"),
    y=alt.Y("sum(no_of_rounds):Q", stack="normalize"),
    color="deal_type",
    tooltip=["deal_type", "no_of_rounds"],
)

# %%
alt
early_vs_late_deals

# %% [markdown]
# ## Overall venture funding

# %% [markdown]
# ### Number of companies

# %%
# Get all unique company IDs
foodtech_ids = [str(s) for s in list(company_to_taxonomy_df.id.unique())]

# %%
# Total number of unique companies
len(foodtech_ids)

# %%
# Total number of companies, excluding agritech
len(
    company_to_taxonomy_df.query(
        "level == 'Category' and Category != 'Agritech'"
    ).id.unique()
)

# %% [markdown]
# ### Total foodtech investment

# %%
# Early stage deals time series
foodtech_ts_early = (
    au.cb_get_all_timeseries(
        DR.company_data.query("id in @foodtech_ids"),
        (
            DR.funding_rounds.query("id in @foodtech_ids").query(
                "`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES"
            )
        ),
        period="year",
        min_year=2010,
        max_year=2022,
    )
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(deal_type="Early")
)

# Late stage deals time series
foodtech_ts_late = (
    au.cb_get_all_timeseries(
        DR.company_data.query("id in @foodtech_ids"),
        (
            DR.funding_rounds.query("id in @foodtech_ids").query(
                "`EACH ROUND TYPE` in @utils.LATE_DEAL_TYPES"
            )
        ),
        period="year",
        min_year=2010,
        max_year=2022,
    )
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(deal_type="Late")
)

# Combined dataframe
foodtech_ts = pd.concat([foodtech_ts_early, foodtech_ts_late], ignore_index=True).drop(
    "time_period", axis=1
)

# %%
# Chart showing both types of deals
horizontal_label = "Year"
horizontal_column = "year"
values_label = "Investment (£ billions)"
values_column = "raised_amount_gbp_total"
tooltip = [
    alt.Tooltip(f"{horizontal_column}:O", title=horizontal_label),
    alt.Tooltip(f"{values_column}:Q", format=",.3f", title=values_column),
]

data_early_late = foodtech_ts.assign(
    raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
).query("year < 2022")

fig = (
    alt.Chart(
        data_early_late,
        width=400,
        height=200,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_column}:O", title=""),
        alt.Y(
            f"{values_column}:Q",
            title=values_label,
        ),
        tooltip=tooltip,
        color=alt.Color(
            "deal_type", sort=["Late", "Early"], legend=alt.Legend(title="Deal type")
        ),
        order=alt.Order(
            # Sort the segments of the bars by this field
            "deal_type",
            sort="ascending",
        ),
    )
)
fig = pu.configure_plots(fig)
fig

# %%
# Chart with only the early stage deals
data_early = data_early_late.query('deal_type == "Early"')

fig = (
    alt.Chart(
        data_early,
        width=400,
        height=200,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_column}:O", title=""),
        alt.Y(
            f"{values_column}:Q",
            title=values_label,
        ),
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig, f"v{VERSION_NAME}_total_early_investment", filetypes=["html", "png"]
)

# %%
# Smoothed growth estimate from 2011 to 2021
au.smoothed_growth(data_early.drop(["deal_type"], axis=1), 2011, 2021)


# %%
# Percentage difference between 2011 and 2021
au.percentage_change(
    data_early.query("`year`==2011")[values_column].iloc[0],
    data_early.query("`year`==2021")[values_column].iloc[0],
)

# %%
# Percentage difference between 2020 and 2021
au.percentage_change(
    data_early.query("`year`==2020")[values_column].iloc[0],
    data_early.query("`year`==2021")[values_column].iloc[0],
)

# %%
# Magnitude and growth between 2017 and 2021
au.estimate_magnitude_growth(data_early.drop(["deal_type"], axis=1), 2017, 2021)

# %%
# Chart with only the early stage deals
data_late = data_early_late.query('deal_type == "Late"')

fig = (
    alt.Chart(
        data_late,
        width=400,
        height=200,
    )
    .mark_bar(color=pu.NESTA_COLOURS[1])
    .encode(
        alt.X(f"{horizontal_column}:O", title=""),
        alt.Y(
            f"{values_column}:Q",
            title=values_label,
        ),
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig, f"v{VERSION_NAME}_total_late_investment", filetypes=["html", "png"]
)

# %%
# Chart showing the number of both types of deals
horizontal_label = "Year"
horizontal_column = "year"
values_label = "Number of rounds"
values_column = "no_of_rounds"
tooltip = [
    alt.Tooltip(f"{horizontal_column}:O", title=horizontal_label),
    alt.Tooltip(f"{values_column}:Q", title=values_column),
]

data_early_late = foodtech_ts.query("year < 2022")

fig = (
    alt.Chart(
        data_early_late,
        width=400,
        height=200,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_column}:O", title=""),
        alt.Y(
            f"{values_column}:Q",
            title=values_label,
        ),
        tooltip=tooltip,
        color=alt.Color(
            "deal_type", sort=["Late", "Early"], legend=alt.Legend(title="Deal type")
        ),
        order=alt.Order(
            # Sort the segments of the bars by this field
            "deal_type",
            sort="ascending",
        ),
    )
)
fig = pu.configure_plots(fig)
fig

# %% [markdown]
# ### UK food tech investment

# %%
country = "United Kingdom"
df_uk = DR.company_data.query("id in @foodtech_ids").query("country == @country").copy()
df_uk_rounds = DR.funding_rounds.query("id in @df_uk.id.to_list()").copy()

# %%
foodtech_ts_early = (
    au.cb_get_all_timeseries(
        df_uk,
        df_uk_rounds.query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES"),
        period="year",
        min_year=2010,
        max_year=2022,
    )
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(deal_type="Early")
)

foodtech_ts_late = (
    au.cb_get_all_timeseries(
        df_uk,
        df_uk_rounds.query("`EACH ROUND TYPE` in @utils.LATE_DEAL_TYPES"),
        period="year",
        min_year=2010,
        max_year=2022,
    )
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(deal_type="Late")
)
foodtech_ts = pd.concat([foodtech_ts_early, foodtech_ts_late])

# %%
horizontal_label = "Year"
values_label = "Investment (bn GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    foodtech_ts.assign(
        raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
    )
    .query("time_period < 2022")
    .rename(
        columns={
            "time_period": horizontal_label,
            "raised_amount_gbp_total": values_label,
        }
    )
)

fig = (
    alt.Chart(
        data.assign(
            **{horizontal_label: pu.convert_time_period(data[horizontal_label], "Y")}
        ),
        width=400,
        height=200,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O"),
        alt.Y(
            f"sum({values_label}):Q",
            title="Raised investment (bn GBP)"
            # scale=alt.Scale(domain=[0, 1200])
            # stack='normalize',
        ),
        tooltip=tooltip,
        color=alt.Color(
            "deal_type",
            sort=["Early", "Late"],
            # legend=None,
            legend=alt.Legend(title="Deal type"),
        ),
        order=alt.Order(
            # Sort the segments of the bars by this field
            "deal_type",
            sort="ascending",
        ),
    )
)
fig = pu.configure_plots(fig)
fig

# %%
au.percentage_change(
    data.query("`Year`==2011 and deal_type == 'Early'")[values_label].iloc[0],
    data.query("`Year`==2021 and deal_type == 'Early'")[values_label].iloc[0],
)

# %%
data.query("`Year`==2021 and deal_type == 'Early'")[values_label].iloc[0] / data.query(
    "`Year`==2011 and deal_type == 'Early'"
)[values_label].iloc[0]

# %%
au.percentage_change(
    data.query("`Year`==2020 and deal_type == 'Early'")[values_label].iloc[0],
    data.query("`Year`==2021 and deal_type == 'Early'")[values_label].iloc[0],
)

# %% [markdown]
# # Early vs late deals per major category

# %%
importlib.reload(utils)

# %%
taxonomy = utils.get_taxonomy_dict(taxonomy_df)

# %%
amounts = []
for cat in list(taxonomy.keys()):
    foodtech_ids_cat = list(
        company_to_taxonomy_df.query("Category == @cat").id.astype(str).unique()
    )
    for deal_type in ["Early", "Late"]:
        if deal_type == "Early":
            deal_types = utils.EARLY_DEAL_TYPES
        else:
            deal_types = utils.LATE_DEAL_TYPES

        amount = (
            DR.funding_rounds.query("id in @foodtech_ids_cat")
            .query("`EACH ROUND TYPE` in @deal_types")
            .query('announced_on > "2016-12-31" and announced_on < "2022-01-01"')
            .raised_amount_gbp.sum()
        )
        amounts.append([cat, deal_type, amount])


df_major_amount_deal_type = pd.DataFrame(
    data=amounts, columns=["Category", "Deal type", "raised_amount_gbp"]
).assign(Investment=lambda df: df.raised_amount_gbp / 1e3)
df_major_amount_deal_type


# %%
category_label = "Category"
values_label = "Investment"

fig = (
    alt.Chart(
        df_major_amount_deal_type,
        width=350,
        height=300,
    )
    .mark_bar()
    .encode(
        alt.X(
            f"{values_label}",
            title="Investment (£ billions)",
            axis=alt.Axis(labelFlush=False),
        ),
        alt.Y(f"{category_label}", sort="-x", title=""),
        color=alt.Color("Deal type"),
        order=alt.Order(
            # Sort the segments of the bars by this field
            "Deal type",
            sort="ascending",
        ),
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig,
    f"v{VERSION_NAME}_total_investment_early_late",
    filetypes=["html", "svg", "png"],
)

# %%
foodtech_ids_cat = list(
    company_to_taxonomy_df.query("Category == 'Logistics'").id.astype(str).unique()
)


# %%
foodtech_ids_cat = list(
    company_to_taxonomy_df.query("Category == 'Health'").id.astype(str).unique()
)

# %%
foodtech_ids_agritech = list(
    company_to_taxonomy_df.query("Category == 'Agritech'").id.astype(str).unique()
)
foodtech_ids_minusAgritech = (
    company_to_taxonomy_df[-company_to_taxonomy_df.id.isin(foodtech_ids_agritech)]
    .id.astype(str)
    .unique()
)

# %%
funding_cat = (
    DR.funding_rounds.query("id in @foodtech_ids_cat")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .query('announced_on > "2016-12-31" and announced_on < "2022-01-01"')
    .raised_amount_gbp.sum()
)

# %%
funding_total = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .query('announced_on > "2016-12-31" and announced_on < "2022-01-01"')
    .raised_amount_gbp.sum()
)

# %%
funding_total_minusAgritech = (
    DR.funding_rounds.query("id in @foodtech_ids_minusAgritech")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .query('announced_on > "2016-12-31" and announced_on < "2022-01-01"')
    .raised_amount_gbp.sum()
)

# %%
funding_cat / funding_total

# %%
funding_cat / funding_total_minusAgritech

# %% [markdown]
# ## Major categories

# %%
importlib.reload(utils)

# %%
# Initialise a Dealroom wrangler instance
importlib.reload(wu)
DR = wu.DealroomWrangler()

# %%
category_ids = utils.get_category_ids(
    taxonomy_df,
    utils.rejected_tags,
    company_to_taxonomy_df,
    DR,
    "Category",
)

# %%
magnitude_vs_growth = utils.get_trends(
    taxonomy_df, utils.rejected_tags, "Category", company_to_taxonomy_df, DR
)
magnitude_vs_growth.sort_values("growth")

# %%
# fig_growth_vs_magnitude(
#     magnitude_vs_growth,
#     colour_field="Major",
#     text_field="Major",
#     horizontal_scale="log",
# ).interactive()

# %%
magnitude_vs_growth_plot = magnitude_vs_growth.assign(
    magnitude=lambda df: df.Magnitude / 1e3
)

# %%
magnitude_vs_growth_plot.magnitude.median()

# %%
from innovation_sweet_spots.utils import chart_trends

# %%
chart_trends._epsilon = 0.05

# %%
magnitude_vs_growth_plot

# %%
fig = chart_trends.mangitude_vs_growth_chart(
    magnitude_vs_growth_plot,
    x_limit=16,
    y_limit=6.5,
    mid_point=1.3,
    baseline_growth=1.28,
    values_label="Average investment per year (£ billions)",
    text_column="Category",
    width=425,
)
fig

# %%
AltairSaver.save(
    fig,
    f"v{VERSION_NAME}_growth_vs_magnitude_Category",
    filetypes=["html", "svg", "png"],
)

# %%
category_ids = utils.get_category_ids(
    taxonomy_df, utils.rejected_tags, company_to_taxonomy_df, DR, "Category"
)
category_ts = utils.get_category_ts(category_ids, DR)

# %%
category_ts.head(2)

# %%
utils.get_estimates(
    category_ts,
    value_column="raised_amount_gbp_total",
    time_column="year",
    category_column="Category",
    estimate_function=au.growth,
    year_start=2019,
    year_end=2020,
)

# %% [markdown]
# #### Time series

# %%
category_ts.head(1)

# %%
# category = "cooking and kitchen"
category = "Retail and restaurants"
# category = "Logistics"
# category = 'innovative food'
# category = 'health'

horizontal_label = "Year"
values_label = "Investment (million GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    category_ts.assign(raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total)
    .query("time_period < 2022")
    .query("Category == @category")
    .rename(
        columns={
            "time_period": horizontal_label,
            "raised_amount_gbp_total": values_label,
        }
    )
)

fig = (
    alt.Chart(
        data.assign(
            **{horizontal_label: pu.convert_time_period(data[horizontal_label], "Y")}
        ),
        width=400,
        height=200,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O"),
        alt.Y(
            f"{values_label}:Q",
            # scale=alt.Scale(domain=[0, 1200])
        ),
        tooltip=tooltip,
    )
)
pu.configure_plots(fig)

# %%
ids = company_to_taxonomy_df.query("Category == @category")
(
    DR.funding_rounds.query("id in @ids.id.to_list()")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .merge(DR.company_data[["id", "NAME", "PROFILE URL", "country"]])
    .groupby(["id", "NAME", "country", "PROFILE URL"], as_index=False)
    .sum()
    .sort_values("raised_amount_gbp", ascending=False)
).head(10)

# %%
DR.funding_rounds.id.iloc[0]

# %%
# fig_category_growth(
#     magnitude_vs_growth_late, colour_field="Major", text_field="Major", height=300
# )

# %% [markdown]
# ## Minor categories (medium granularity)

# %%
importlib.reload(utils)

# %%
taxonomy_df.loc[25, "Category"] = "Innovative food"
taxonomy_df.loc[25, "Sub Category"] = "Alt protein (all)"

# %%
alt_protein_cats = [
    "Plant-based",
    "Fermentation",
    "Lab meat",
    "Insects",
    "Alt protein (other)",
]

# %%
df = (
    company_to_taxonomy_df[company_to_taxonomy_df.Category.isin(alt_protein_cats)]
    .drop_duplicates("id")
    .copy()
)
df.loc[:, "Category"] = "Alt protein (all)"

company_to_taxonomy_df_ = pd.concat([company_to_taxonomy_df, df], ignore_index=True)

# %%
magnitude_vs_growth_minor = (
    utils.get_trends(
        taxonomy_df, utils.rejected_tags, "Sub Category", company_to_taxonomy_df_, DR
    )
    .query("`Category` != 'Agritech'")
    .query("`Sub Category` != 'Alt protein (other)'")
)
magnitude_vs_growth_minor.sort_values("growth")

# %%
major_sort_order = magnitude_vs_growth.sort_values("Growth").Category.to_list()
major_sort_order

# %%
major_sort_order = magnitude_vs_growth.sort_values("Growth").Category.to_list()
data = magnitude_vs_growth_minor.copy()
data["Category"] = pd.Categorical(data["Category"], categories=major_sort_order)
data = data.sort_values(["Category", "growth"], ascending=False)
data = (
    data.assign(Increase=lambda df: df.growth > 0)
    .assign(Magnitude_log=lambda df: np.log10(df.Magnitude))
    .assign(Magnitude=lambda df: df.Magnitude / 1e3)
)

# %%
colour_field = "Category"
text_field = "Sub Category"
height = 500

# Chart
fig = (
    alt.Chart(
        data,
        width=500,
        height=height,
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
                labelFlush=False,
            ),
            scale=alt.Scale(domain=(-1, 100)),
        ),
        y=alt.Y(
            "Sub Category:N",
            sort=data["Sub Category"].to_list(),
            # axis=alt.Axis(title="", labels=False),
            axis=None,
        ),
        size=alt.Size(
            "Magnitude",
            title="Avg yearly investment (£ bn)",
            legend=alt.Legend(orient="left"),
            scale=alt.Scale(domain=[0.1, 4]),
        ),
        color=alt.Color(colour_field, legend=alt.Legend(orient="left")),
        # size="cluster_size:Q",
        #         color=alt.Color(f"{colour_title}:N", legend=None),
        tooltip=[
            alt.Tooltip("Category:N", title="Category"),
            alt.Tooltip(
                "Magnitude:Q",
                format=",.3f",
                title="Average yearly investment (billion GBP)",
            ),
            "Number of companies",
            "Number of deals",
            alt.Tooltip("growth:Q", format=",.0%", title="Growth"),
        ],
    )
)

# Text labels
text = (
    alt.Chart(data)
    .mark_text(align="left", baseline="middle", font=pu.FONT, dx=7, fontSize=14)
    .encode(
        text=text_field,
        x="growth:Q",
        y=alt.Y("Sub Category:N", sort=data["Sub Category"].to_list(), title=""),
    )
)

# Baseline
baseline_rule = (
    alt.Chart(pd.DataFrame({"x": [1.28]}))
    .mark_rule(strokeDash=[5, 7], size=1, color="k")
    .encode(
        x=alt.X(
            "x:Q",
        )
    )
)

final_fig = pu.configure_titles(pu.configure_axes((baseline_rule + fig + text)), "", "")
final_fig

# %%
AltairSaver.save(
    final_fig, f"v{VERSION_NAME}_growth_SubCategories", filetypes=["html", "svg", "png"]
)


# %%
pd.set_option("max_colwidth", 200)
category = "Reformulation"
ids = company_to_taxonomy_df.query("Category == @category")
(
    DR.company_data[["id", "NAME", "TAGLINE", "PROFILE URL", "WEBSITE", "country"]]
    .query("id in @ids.id.to_list()")
    .merge(
        (
            DR.funding_rounds.query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
            .groupby(["id"], as_index=False)
            .sum()
        ),
        on="id",
        how="left",
    )
    .sort_values("raised_amount_gbp", ascending=False)
    .drop(["funding_round_id", "raised_amount_usd"], axis=1)
)

# %% [markdown]
# ## Sub category time series

# %%
# Get company ids for each category
subcategory_ids = utils.get_category_ids(
    taxonomy_df,
    utils.rejected_tags,
    company_to_taxonomy_df,
    DR,
    "Sub Category",
)
# Get time series for each category
variable = "raised_amount_gbp_total"

subcategory_ts = (
    utils.get_category_ts(subcategory_ids, DR)
    .rename(columns={"Category": "Sub Category"})
    .query("year < 2022")
)


# %%
subcategory_ts.head(1)

# %%
au.moving_average(subcategory_ts.query("`Sub Category` == 'Lab meat'"))

# %%
(367.165106 - 5.722942) / 5.722942

# %%
367.165106 / 5.722942

# %%
importlib.reload(pu)

# %%
cats = ["Kitchen tech", "Dark kitchen"]

fig = pu.configure_plots(
    pu.ts_smooth(
        subcategory_ts.assign(**{variable: lambda df: df[variable] / 1}),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ million)",
        category_column="Sub Category",
        amount_div=1,
    )
)
fig

# %%
AltairSaver.save(
    fig, f"v{VERSION_NAME}_ts_SubCategory_cooking", filetypes=["html", "svg", "png"]
)


# %%
cats = ["Lab meat", "Insects", "Plant-based", "Fermentation"]

fig = pu.configure_plots(
    pu.ts_smooth(
        subcategory_ts.assign(**{variable: lambda df: df[variable] / 1000}),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ billion)",
        category_column="Sub Category",
        amount_div=1,
    )
)
fig

# %%
AltairSaver.save(
    fig, f"v{VERSION_NAME}_ts_SubCategory_alt_protein", filetypes=["html", "svg", "png"]
)


# %%
cats = ["Innovative food (other)", "Reformulation"]

fig = pu.configure_plots(
    pu.ts_smooth(
        subcategory_ts.assign(**{variable: lambda df: df[variable] / 1000}),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ billion)",
        category_column="Sub Category",
        amount_div=1,
    )
)
fig

# %%
AltairSaver.save(
    fig,
    f"v{VERSION_NAME}_ts_SubCategory_innovative_food",
    filetypes=["html", "svg", "png"],
)


# %%
cats = ["Biomedical", "Personalised nutrition"]

fig = pu.configure_plots(
    pu.ts_smooth(
        subcategory_ts.assign(**{variable: lambda df: df[variable]}),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ million)",
        category_column="Sub Category",
        amount_div=1,
    )
)
fig

# %%
AltairSaver.save(
    fig, f"v{VERSION_NAME}_ts_SubCategory_health", filetypes=["html", "svg", "png"]
)

# %%
cats = ["Waste reduction", "Packaging"]

fig = pu.configure_plots(
    pu.ts_smooth(
        subcategory_ts.assign(**{variable: lambda df: df[variable]}),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ million)",
        category_column="Sub Category",
        amount_div=1,
    )
)
fig

# %%
AltairSaver.save(
    fig, f"v{VERSION_NAME}_ts_SubCategory_Food_waste", filetypes=["html", "svg", "png"]
)

# %%
cats = ["Retail", "Restaurants"]

fig = pu.configure_plots(
    pu.ts_smooth(
        subcategory_ts.assign(**{variable: lambda df: df[variable]}),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ million)",
        category_column="Sub Category",
        amount_div=1,
    )
)
fig

# %%
AltairSaver.save(
    fig,
    f"v{VERSION_NAME}_ts_SubCategory_Retail_restaurant",
    filetypes=["html", "svg", "png"],
)

# %%
pd.set_option("max_colwidth", 200)
# category = "reformulation"
ids = company_to_taxonomy_df.query("Category == @category")
(
    DR.company_data[["id", "NAME", "TAGLINE", "PROFILE URL", "WEBSITE", "country"]]
    .query("id in @ids.id.to_list()")
    .merge(
        (
            DR.funding_rounds.query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
            .groupby(["id"], as_index=False)
            .sum()
        ),
        on="id",
        how="left",
    )
    .sort_values("raised_amount_gbp", ascending=False)
    .drop(["funding_round_id", "raised_amount_usd"], axis=1)
)

# %% [markdown]
# ## Exporting the list of startups

# %%
pd.set_option("max_colwidth", 200)
# category = "reformulation"
ids = company_to_taxonomy_df
df_export = (
    company_to_taxonomy_df.query("level == 'Minor'")
    .merge(
        DR.company_data[["id", "NAME", "TAGLINE", "PROFILE URL", "WEBSITE", "country"]],
        on="id",
        how="left",
    )
    .merge(
        (
            DR.funding_rounds.query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
            .groupby(["id"], as_index=False)
            .sum()
        ),
        on="id",
        how="left",
    )
    .sort_values("raised_amount_gbp", ascending=False)
    .drop(["funding_round_id", "raised_amount_usd", "level"], axis=1)
    .merge(taxonomy_df[["Minor", "Major"]], left_on="Category", right_on="Minor")
    .rename(columns={"Category": "sub_category", "Major": "category"})
    .sort_values(["raised_amount_gbp"], ascending=False)
    .sort_values(["category", "sub_category"])
)[
    [
        "id",
        "NAME",
        "TAGLINE",
        "PROFILE URL",
        "WEBSITE",
        "country",
        "raised_amount_gbp",
        "category",
        "sub_category",
    ]
]

df_export.loc[df_export.category == "agritech", "sub_category"] = "-"

# %%
df_export.to_csv(
    PROJECT_DIR / "outputs/foodtech/interim/foodtech_reviewed_VC_final_v2022_08_31.csv",
    index=False,
)

# %% [markdown]
# # Country performance

# %%
DR.funding_rounds.head(1)

# %%
data = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(DR.company_data[["id", "country"]])
    .merge(company_to_taxonomy_df.query("level == 'Category'"))
    .groupby(["country"], as_index=False)
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"))
    .assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1000)
    .sort_values("raised_amount_gbp", ascending=False)
)
data.head(10)

# %%
data_countries_early = data.copy()

# %%
data.query("country in @EU_countries").sum()

# %%
fig = (
    alt.Chart(
        data.head(10),
        width=200,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.Y(f"country:N", sort="-x", title=""),
        alt.X(
            f"raised_amount_gbp:Q",
            title="Investment amount (bn GBP)"
            # scale=alt.Scale(domain=[0, 1200])
        ),
        tooltip=["country", "raised_amount_gbp"],
    )
)
fig = pu.configure_plots(
    fig, "Investment amounts by country", "Early stage deals, 2017-2021"
)
fig

# %%
AltairSaver.save(fig, f"vJuly18_countries_early", filetypes=["html", "png"])

# %%
data = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.LATE_DEAL_TYPES")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(DR.company_data[["id", "country"]])
    .merge(company_to_taxonomy_df.query("level == 'Category'"))
    .groupby(["country"], as_index=False)
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"))
    .assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1000)
    .sort_values("raised_amount_gbp", ascending=False)
)
data.head(10)

# %%
data.query("country in @EU_countries").sum()

# %%
fig = (
    alt.Chart(
        data.head(10),
        width=200,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[1])
    .encode(
        alt.Y(f"country:N", sort="-x", title="Country"),
        alt.X(
            f"raised_amount_gbp:Q",
            title="Investment amount (bn GBP)"
            # scale=alt.Scale(domain=[0, 1200])
        ),
        tooltip=["country", "raised_amount_gbp"],
    )
)
fig = pu.configure_plots(
    fig, "Investment amounts by country", "Late stage deals, 2017-2021"
)
fig

# %%
AltairSaver.save(fig, f"vJuly18_countries_late", filetypes=["html", "png"])

# %%
# countries = ['United Kingdom', 'United States', 'China', 'India', 'Singapore', 'Germany', 'France']
data = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(DR.company_data[["id", "country"]])
    .merge(company_to_taxonomy_df.query("level == 'Category'"))
    .groupby(["country", "Category"], as_index=False)
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"))
    .assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1000)
    # .query("country in @countries")
    # .sort_values('raised_amount_gbp', ascending=False)
)
data_total = data.groupby(["country"], as_index=False).agg(
    total_amount=("raised_amount_gbp", "sum")
)

data = (
    data.merge(data_total, on=["country"], how="left").assign(
        percentage=lambda df: df.raised_amount_gbp / df.total_amount
    )
    # .query("country in @countries")
)

cats = [
    "Logistics",
    "Innovative food",
    "Health",
    "Retail and restaurants",
    "Cooking and kitchen",
    "Food waste",
    "Agritech",
]
data_final = []
for cat in cats:
    data_final.append(
        data.copy()
        .query("Category == @cat")
        .sort_values("raised_amount_gbp", ascending=False)
        .head(8)
    )
data = pd.concat(data_final, ignore_index=True)


# %%
# data.query("Category == 'innovative food'").sort_values("raised_amount_gbp", ascending=False).head(8)

# %%
df_check = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(DR.company_data[["id", "NAME", "country"]])
    .merge(company_to_taxonomy_df.query("level == 'Category'"))
    # .groupby(['country', 'Category'], as_index=False)
    # .agg(raised_amount_gbp=('raised_amount_gbp', 'sum'))
    .assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1000)
    # .query('country== "United Kingdom"')
    # .query('Category== "cooking and kitchen"')
)

# %%
# df_check

# %%
fig = (
    alt.Chart((data.query("Category in @cats")))
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        # x=alt.X('percentage:Q', title="", axis=alt.Axis(format='%')),
        x=alt.X("raised_amount_gbp:Q", title="Investment (bn. GBP)"),
        # y=alt.Y('country:N', sort=countries, title=""),
        y=alt.Y("country:N", sort="-x", title=""),
        # color='country:N',
        facet=alt.Facet(
            "Category:N", title="", columns=2, header=alt.Header(labelFontSize=14)
        ),
        tooltip=[alt.Tooltip("raised_amount_gbp", format=".3f")],
    )
    .properties(
        width=180,
        height=180,
    )
    .resolve_scale(x="independent", y="independent")
)


fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig, f"v{VERSION_NAME}_countries_major_early", filetypes=["html", "png"]
)

# %%
# (
#     DR.company_data.merge(company_to_taxonomy_df)
#     .query("Category=='health'")
#     .query("country=='United Kingdom'")
#     .sort_values("TOTAL FUNDING (EUR M)", ascending=False)
# )[["NAME", "PROFILE URL", "TOTAL FUNDING (EUR M)"]].sum()

# %%
# (
#     DR.funding_rounds.merge(company_to_taxonomy_df)
#     .query(
#         "`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES or `EACH ROUND TYPE` in @utils.LATE_DEAL_TYPES"
#     )
#     .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
#     .merge(DR.company_data[["id", "NAME", "PROFILE URL", "country"]])
#     .query("Category=='health'")
#     .query("country=='United Kingdom'")
#     .groupby(["NAME", "PROFILE URL"], as_index=False)
#     .sum()
#     .sort_values("raised_amount_gbp", ascending=False)
# )[["NAME", "PROFILE URL", "raised_amount_gbp"]].iloc[1:].sum()

# %%
# (
#     DR.funding_rounds.merge(company_to_taxonomy_df)
#     .query(
#         "`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES or `EACH ROUND TYPE` in @utils.LATE_DEAL_TYPES"
#     )
#     .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
#     .merge(DR.company_data[["id", "NAME", "PROFILE URL", "country"]])
#     .query("Category=='logistics'")
#     # .query("country=='United Kingdom'")
#     .groupby(["NAME", "PROFILE URL"], as_index=False)
#     .sum()
#     .sort_values("raised_amount_gbp", ascending=False)
# )[["NAME", "PROFILE URL", "raised_amount_gbp"]]

# %% [markdown]
# ## Growth by countries

# %%
country = "United Kingdom"

# %%
countries = data_countries_early.country.head(10).to_list()

# %%
growth = []
for country in countries:
    df_companies = DR.company_data.query("id in @foodtech_ids").query(
        "country == @country"
    )
    df_companies_id = df_companies.id.astype(str).to_list()

    country_ts_early = (
        au.cb_get_all_timeseries(
            df_companies,
            (
                DR.funding_rounds.query("id in @df_companies_id").query(
                    "`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES"
                )
            ),
            period="year",
            min_year=2010,
            max_year=2022,
        )
        .assign(year=lambda df: df.time_period.dt.year)
        .assign(deal_type="Early")
    )
    growth.append(
        au.smoothed_growth(
            country_ts_early.query("deal_type == 'Early'").drop(
                ["time_period"], axis=1
            ),
            2017,
            2021,
        ).raised_amount_gbp_total
    )


# %%
countries_growth = pd.DataFrame(
    {
        "country": countries,
        "growth": growth,
    }
)

# %%
countries_growth.sort_values("growth")

# %%
countries_growth_magnitude = countries_growth.merge(
    data_countries_early, how="left"
).assign(
    growth=lambda df: df.growth / 100,
    magnitude=lambda df: df.raised_amount_gbp,
)
countries_growth_magnitude.sort_values("growth")

# %%
countries_growth_magnitude.raised_amount_gbp.median()

# %%
fig = chart_trends.mangitude_vs_growth_chart(
    countries_growth_magnitude,
    x_limit=45,
    y_limit=20,
    mid_point=3.6,
    baseline_growth=1.28,
    values_label="Investment (£ billions)",
    text_column="country",
    width=425,
)
fig.interactive()

# %%
fig = (
    alt.Chart(
        countries_growth_magnitude,
        width=200,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.Y(f"country:N", sort="-x", title=""),
        alt.X(
            f"growth:Q",
            title="Growth",
            axis=alt.Axis(format="%"),
            # scale=alt.Scale(domain=[0, 1200])
        ),
        # tooltip=["country", "raised_amount_gbp"],
    )
)
fig = pu.configure_plots(fig)
fig


# %%

# %% [markdown]
# # Maturation of companies

# %%
def deal_type(dealtype):
    if dealtype in utils.EARLY_DEAL_TYPES:
        return "Early"
    elif dealtype in utils.LATE_DEAL_TYPES:
        return "Late"
    else:
        return "n/a"


# %%
importlib.reload(utils)


# %%
def deal_type_no(dealtype):
    for i, d in enumerate(pu.DEAL_CATEGORIES):
        if dealtype == d:
            return i
    return 0


# %%
data = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(company_to_taxonomy_df.query("level == 'Major'"), how="left")
    .copy()
    .assign(
        deal_type=lambda df: df["raised_amount_gbp"].apply(
            lambda x: utils.deal_amount_to_range(x * 1000)
        )
    )
    .assign(deal_type_=lambda df: df["deal_type"].apply(deal_type_no))
    .groupby(["Category", "deal_type", "deal_type_"], as_index=False)
    .agg(counts=("id", "count"), amounts=("raised_amount_gbp", "sum"))
)

# %%
# data

# %%
fig = (
    alt.Chart(data.query("deal_type != 'n/a'"))
    .mark_bar()
    .encode(
        # x=alt.X('percentage:Q', title="", axis=alt.Axis(format='%')),
        # y= alt.X("sum(amounts):Q", title="Number of deals"),#, stack="normalize", axis=alt.Axis(format="%")),
        y=alt.X(
            "sum(amounts):Q", title="", stack="normalize", axis=alt.Axis(format="%")
        ),
        # y=alt.Y('country:N', sort=countries, title=""),
        x=alt.Y(
            "Category:N",
            title="",
            axis=alt.Axis(labelAngle=-45),
            sort=[
                "logistics",
                "food waste",
                "retail and restaurants",
                "cooking and kitchen",
                "agritech",
                "innovative food",
                "health",
            ],
        ),
        color=alt.Color("deal_type", title="Deal size", sort=pu.DEAL_CATEGORIES),
        order=alt.Order(
            # Sort the segments of the bars by this field
            "deal_type_",
            sort="ascending",
        ),
        # color='country:N',
        # facet=alt.Facet('Category:N',title="", columns=3, header=alt.Header(labelFontSize=14)),
        tooltip=["deal_type", "counts", "sum(counts)"],
    )
    .properties(
        width=400,
        height=300,
    )
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(fig, f"v{VERSION_NAME}_deal_types_not_norm", filetypes=["html", "png"])

# %% [markdown]
# ## Acquisitions

# %%
data = (
    DR.funding_rounds.query("id in @foodtech_ids")
    # .query("`EACH ROUND TYPE` == 'ACQUISITION'")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'").merge(
        company_to_taxonomy_df.query("level == 'Major'"), how="left"
    )
    # .copy()
    # .assign(deal_type=lambda df: df["raised_amount_gbp"].apply(lambda x: utils.deal_amount_to_range(x*1000)))
    # .assign(deal_type_=lambda df: df["deal_type"].apply(deal_type_no))
    # .groupby(['Category'], as_index=False).agg(counts=("id", "count"))
)

# %%
comp_names = [
    "Nestlé",
    "PepsiCo",
    "Unilever",
    "Tyson Ventures",
    "Archer Daniels Midland",
    "Cargill",
    "Danone",
    # 'Associated British Foods'
]
# comp_name = 'Nestlé' # meal kits and pet food
# comp_name = 'PepsiCo'
# comp_name = 'Unilever N.V.'
# comp_name = 'Delivery Hero'
# comp_name = 'Bite Squad'
# comp_name = 'Anheuser Busch InBev'
# comp_name = 'Zomato'
# comp_name = "The Middleby Corp"
# comp_name = 'Constellation Brands'
# comp_name = "Foodpanda"
# comp_name = "Syngenta"
# comp_name = 'Danone'
pd.set_option("max_colwidth", 200)
dfs = []
for comp_name in comp_names:
    dfs.append(
        data[
            (data["EACH ROUND INVESTORS"].isnull() == False)
            & data["EACH ROUND INVESTORS"].str.contains(comp_name)
        ].merge(DR.company_data[["id", "NAME", "PROFILE URL", "country"]])
    )


# %%
comp_names_ = [
    "Nestlé",
    "PepsiCo",
    "Unilever N.V.",
    "Tyson Ventures",
    "Archer Daniels Midland Company",
    "Cargill",
    "Danone",
]

# %%
[len(df.NAME.unique()) for df in dfs]


# %%
dfs[2].query("`EACH ROUND TYPE` == 'ACQUISITION'")[["NAME"]]

# %%
pd.DataFrame(
    data={
        "company": comp_names_,
        "investments": [str(list(df.NAME.unique())) for df in dfs],
    }
)


# %%
(
    data.groupby("EACH ROUND INVESTORS", as_index=False)
    .agg(counts=("id", "count"))
    .sort_values("counts", ascending=False)
).reset_index().iloc[1:11]

# %%
data = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` == 'ACQUISITION'")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(company_to_taxonomy_df.query("level == 'Major'"), how="left")
    # .copy()
    # .assign(deal_type=lambda df: df["raised_amount_gbp"].apply(lambda x: utils.deal_amount_to_range(x*1000)))
    # .assign(deal_type_=lambda df: df["deal_type"].apply(deal_type_no))
    # .groupby(['Category'], as_index=False).agg(counts=("id", "count"))
)

# %%
category_ids = get_category_ids(taxonomy_df, rejected_tags, DR, "Major")
category_ts = get_category_ts(category_ids, DR, ["ACQUISITION"])

# %%
magnitude_vs_growth = get_trends(
    taxonomy_df, rejected_tags, "Major", DR, ["ACQUISITION"]
)
magnitude_vs_growth

# %%
category_ts.query("Category == 'innovative food'")

# %%
# data.groupby('EACH ROUND INVESTORS').agg(counts=('id', 'count')).sort_values('counts', ascending=False).head(20)

# %%
# comp_name = 'Nestlé' # meal kits and pet food
# comp_name = 'PepsiCo'
# comp_name = 'Unilever N.V.'
# comp_name = 'Delivery Hero'
# comp_name = 'Bite Squad'
# comp_name = 'Anheuser Busch InBev'
# comp_name = 'Zomato'
# comp_name = "The Middleby Corp"
comp_name = "Constellation Brands"
# comp_name = "Foodpanda"
# comp_name = "Syngenta"
comp_name = "Archer"

(
    data[
        (data["EACH ROUND INVESTORS"].isnull() == False)
        & data["EACH ROUND INVESTORS"].str.contains(comp_name)
    ].merge(DR.company_data[["id", "NAME", "PROFILE URL", "country"]])
)[
    [
        "id",
        "NAME",
        "PROFILE URL",
        "announced_on",
        "Category",
        "EACH ROUND INVESTORS",
        "raised_amount_gbp",
        "country",
    ]
]

# %%
# comp_name = 'Nestlé'
# comp_name = 'PepsiCo'
# comp_name = 'Unilever N.V.'
# comp_name = 'Delivery Hero'
# comp_name = 'Bite Squad'
# comp_name = 'Anheuser Busch InBev'
# comp_name = 'Zomato'
(
    data[
        (data["EACH ROUND INVESTORS"].isnull() == False)
        & data["EACH ROUND INVESTORS"].str.contains(comp_name)
    ].merge(DR.company_data[["id", "NAME", "PROFILE URL", "country"]])
)[
    [
        "id",
        "NAME",
        "PROFILE URL",
        "announced_on",
        "Category",
        "EACH ROUND INVESTORS",
        "raised_amount_gbp",
        "country",
    ]
]

# %%
df = (
    DR.funding_rounds.query("id in @foodtech_ids")
    # .query("`EACH ROUND TYPE` == @utils.EARLY_DEAL_TYPES")
    .query("announced_on >= '2010-01-01' and announced_on < '2022-01-01'").merge(
        company_to_taxonomy_df.query("level == 'Major'"), how="left"
    )
)

(
    df[
        (df["EACH ROUND INVESTORS"].isnull() == False)
        & df["EACH ROUND INVESTORS"].str.contains(comp_name)
    ].merge(DR.company_data[["id", "NAME", "TAGLINE", "PROFILE URL"]])
)[
    [
        "id",
        "NAME",
        "TAGLINE",
        "PROFILE URL",
        "announced_on",
        "Category",
        "EACH ROUND TYPE",
        "EACH ROUND INVESTORS",
        "raised_amount_gbp",
    ]
]
