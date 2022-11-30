# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Food tech: Venture funding trends [UK version]
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
from innovation_sweet_spots.utils import chart_trends

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
VERSION_NAME = "Report_VC"

# %%
# Initialise a Dealroom wrangler instance
DR = wu.DealroomWrangler()

# Check the number of companies
len(DR.company_data)

# %% [markdown]
# ### Import reviewed data

# %%
# Check companies with rejected tags
rejected_ids = [
    DR.get_ids_by_labels(row.Category, row.label_type)
    for i, row in DR.labels.query("Category in @utils.rejected_tags").iterrows()
]
rejected_ids = set(itertools.chain(*rejected_ids))

# %%
# Taxonomy file for the VC data
taxonomy_df = pd.read_csv(output_folder / "vc_tech_taxonomy.csv")
# Mapping from minor sub categories to major categories
minor_to_major = load_json(output_folder / "vc_tech_taxonomy_minor_to_major.json")
# Mapping from companies to taxonomy categories
company_to_taxonomy_df = (
    pd.read_csv(output_folder / "vc_company_to_taxonomy.csv")
    .astype({"id": str})
    .query("id not in @rejected_ids")
    .copy()
)

# %%
## Adapting taxonomy to also include a combined alt protein category
# Existing alt protein categories
alt_protein_cats = [
    "Plant-based",
    "Fermentation",
    "Lab meat",
    "Insects",
    "Alt protein (other)",
]
# Combined category
combined_category_name = "Alt protein (all)"

# Adding the combined category to taxonomy dataframe
last_row = len(taxonomy_df)
taxonomy_df.loc[last_row, "Category"] = "Innovative food"
taxonomy_df.loc[last_row, "Sub Category"] = combined_category_name

# Adding the extra mappings to the main table
df = (
    company_to_taxonomy_df[company_to_taxonomy_df.Category.isin(alt_protein_cats)]
    .drop_duplicates("id")
    .copy()
)
df.loc[:, "Category"] = combined_category_name
company_to_taxonomy_df = pd.concat([company_to_taxonomy_df, df], ignore_index=True)


# %%
taxonomy_df

# %%
# Uncomment if doing the analysis soley for the UK
uk_ids = DR.company_data.query("country == 'United Kingdom'").id.to_list()
company_to_taxonomy_df = company_to_taxonomy_df.query("id in @uk_ids")
VERSION_NAME = "October_VC_UK"

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

# %% [markdown]
# ## Overall venture funding

# %% [markdown]
# ### Number of companies

# %%
# Get all unique company IDs
foodtech_ids = set(list(company_to_taxonomy_df.id.unique()))

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
values_label = "Investment (£ millions)"
values_column = "raised_amount_gbp_total"
tooltip = [
    alt.Tooltip(f"{horizontal_column}:O", title=horizontal_label),
    alt.Tooltip(f"{values_column}:Q", format=",.3f", title=values_column),
]

data_early_late = foodtech_ts.assign(
    raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total
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
        tooltip=tooltip,
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
# Chart with only the late stage deals
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
# Chart showing the number of both types of deals (number of deals)
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
# ## Major innovation categories
#
# ### Early vs late deals

# %%
# Create taxonomy dictionary
taxonomy = utils.get_taxonomy_dict(taxonomy_df)
# Iterate through categories and deal types
amounts = []
for cat in list(taxonomy.keys()):
    # Fetch companies in the category
    foodtech_ids_cat = list(
        company_to_taxonomy_df.query("Category == @cat").id.unique()
    )
    # Check for each deal type
    for deal_type in ["Early", "Late"]:
        if deal_type == "Early":
            deal_types = utils.EARLY_DEAL_TYPES
        else:
            deal_types = utils.LATE_DEAL_TYPES

        amount = (
            DR.funding_rounds.query("id in @foodtech_ids_cat")
            # Select only appopriate deals
            .query("`EACH ROUND TYPE` in @deal_types")
            # Select only between 2017 and 2021
            .query(
                'announced_on > "2016-12-31" and announced_on < "2022-01-01"'
            ).raised_amount_gbp.sum()
        )
        # Save the results in a nestaed list
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


# %% [markdown]
# ### Category proportion of total investment

# %%
def get_total_funding(
    ids, deal_types=utils.EARLY_DEAL_TYPES, min_year=2017, max_year=2021
):
    """Sum up total funding across all provided company ids, between 2017 and 2021"""
    return (
        DR.funding_rounds.query("id in @ids")
        .query("`EACH ROUND TYPE` in @deal_types")
        .query(
            f'announced_on > "{min_year-1}-12-31" and announced_on < "{max_year+1}-01-01"'
        )
        .raised_amount_gbp.sum()
    )


# %%
# Get ids for each category
category_ids = utils.get_category_ids(
    taxonomy_df,
    utils.rejected_tags,
    company_to_taxonomy_df,
    DR,
    "Category",
)

# %%
# Total funding in 2017-2021
funding_total = get_total_funding(foodtech_ids, min_year=2017, max_year=2021)

# Total funding in 2017-2021, excluding agritech
foodtech_ids_minusAgritech = company_to_taxonomy_df[
    -company_to_taxonomy_df.id.isin(category_ids["Agritech"])
].id.unique()
funding_total_minusAgritech = get_total_funding(foodtech_ids_minusAgritech)


# %%
print(
    get_total_funding(category_ids["Health"], min_year=2017, max_year=2021)
    / funding_total
)
print(
    get_total_funding(category_ids["Health"], min_year=2017, max_year=2021)
    / funding_total_minusAgritech
)

# %%
print(
    get_total_funding(category_ids["Logistics"], min_year=2017, max_year=2021)
    / funding_total
)
print(
    get_total_funding(category_ids["Logistics"], min_year=2017, max_year=2021)
    / funding_total_minusAgritech
)


# %% [markdown]
# ### Trends analysis

# %%
def _estimate_trend_type(magnitude, growth, mid_point, tolerance=0.1):
    # Flags to double check trend type if ambiguous
    flag_1 = "*" if (abs(magnitude - mid_point) / mid_point) < tolerance else ""
    flag_2 = "*" if abs(growth) < tolerance else ""
    flags = flag_1 + flag_2

    if (growth > 0) and (magnitude < mid_point):
        return "emerging" + flags
    if (growth > 0) and (magnitude >= mid_point):
        return "hot" + flags
    if (growth < 0) and (magnitude < mid_point):
        return "dormant" + flags
    if (growth < 0) and (magnitude >= mid_point):
        return "stable" + flags
    else:
        return "n/a"


def estimate_trend_type(
    mangitude_vs_growth,
    mid_point=None,
    magnitude_column="Magnitude",
    growth_column="growth",
):
    """Suggests the trend type provided the magnitude and growth values"""
    mid_point = mangitude_vs_growth[magnitude_column].median()
    trend_types = []
    for i, row in magnitude_vs_growth.iterrows():
        trend_types.append(
            _estimate_trend_type(row[magnitude_column], row[growth_column], mid_point)
        )
    return mangitude_vs_growth.copy().assign(trend_type_suggestion=trend_types)


# %%
# Smoothed growth estimate of total venture funding 2017-2021, inferred from Crunchbase data
BASELINE_GROWTH = 1.64

# %%
# Caculate mangitude vs growth plots
magnitude_vs_growth = utils.get_trends(
    taxonomy_df, utils.rejected_tags, "Category", company_to_taxonomy_df, DR
)
magnitude_vs_growth_plot = magnitude_vs_growth.assign(
    magnitude=lambda df: df.Magnitude / 1e3
)

magnitude_vs_growth.sort_values("growth")

# %%
order = magnitude_vs_growth.sort_values("Magnitude", ascending=False).Category.to_list()

fig_1 = pu.configure_plots(
    alt.Chart(magnitude_vs_growth, height=200)
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        y=alt.Y("Category", sort=order, title=""),
        x=alt.X(
            "Magnitude",
            title="Yearly average investment (£ millions)",
            axis=alt.Axis(labelFlush=False),
        ),
        tooltip=[
            "Category",
            alt.Tooltip(
                "Magnitude",
                format=".1f",
                title="Yearly average investment (£ millions)",
            ),
        ],
    )
)

fig_2_bars = (
    alt.Chart(magnitude_vs_growth, height=200)
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        y=alt.Y("Category", sort=order, title=""),
        x=alt.X(
            "growth",
            title="Growth",
            axis=alt.Axis(labelFlush=False, format="%"),
        ),
        tooltip=[
            "Category",
            alt.Tooltip("growth", format=".0%", title="Growth"),
        ],
    )
)

fig_2_rule = (
    alt.Chart(pd.DataFrame({"x": [1.64]}))
    .mark_rule(strokeDash=[5, 7], size=1)
    .encode(x="x:Q")
)

fig_2 = pu.configure_plots(fig_2_rule + fig_2_bars)

# %%
fig_1

# %%
fig_2

# %%
AltairSaver.save(
    fig_1,
    f"v{VERSION_NAME}_total_investment_categories",
    filetypes=["html", "svg", "png"],
)

# %%
AltairSaver.save(
    fig_2,
    f"v{VERSION_NAME}_total_investment_categories_growth",
    filetypes=["html", "svg", "png"],
)

# %%
# Chart configs
# mid point
mid_point = magnitude_vs_growth_plot.magnitude.median()
# color gradient width
chart_trends._epsilon = 0.05

fig = chart_trends.mangitude_vs_growth_chart(
    magnitude_vs_growth_plot,
    x_limit=0.4,
    y_limit=6.5,
    mid_point=mid_point,
    baseline_growth=BASELINE_GROWTH,
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

# %% [markdown]
# ### Major category time series
#
# Only for checking, not included in the report

# %%
category_ids = utils.get_category_ids(
    taxonomy_df, utils.rejected_tags, company_to_taxonomy_df, DR, "Category"
)
category_ts = utils.get_category_ts(category_ids, DR)

# %%
category = "Retail and restaurants"
# category = "Logistics"

horizontal_label = "Year"
horizontal_column = "year"
values_label = "Investment (£ billions)"
values_column = "raised_amount_gbp_total"
tooltip = [
    alt.Tooltip(f"{horizontal_column}:O", title=horizontal_label),
    alt.Tooltip(f"{values_column}:Q", format=",.3f", title=values_label),
]

data = (
    category_ts.assign(
        raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
    )
    .query("year < 2022")
    .query("Category == @category")
)

fig = (
    alt.Chart(
        data,
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
    )
)
fig = pu.configure_plots(fig)
fig

# %%
# Check companies that have raised the most money in this category
ids = company_to_taxonomy_df.query("Category == @category")
(
    DR.funding_rounds.query("id in @ids.id.to_list()")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .merge(DR.company_data[["id", "NAME", "PROFILE URL", "country"]])
    .groupby(["id", "NAME", "country", "PROFILE URL"], as_index=False)
    .sum()
    .sort_values("raised_amount_gbp", ascending=False)
).head(10)

# %% [markdown]
# ### Export major category results

# %%
# Magnitude vs growth data
estimate_trend_type(magnitude_vs_growth).to_csv(
    PROJECT_DIR
    / f"outputs/foodtech/trends/venture_capital_{VERSION_NAME}_Category.csv",
    index=False,
)


# %% [markdown]
# ## Minor innovation subcategories

# %% [markdown]
# ### Trends analysis

# %%
# Calculate magnitude and growth for the innovation subcategories
magnitude_vs_growth_minor = (
    utils.get_trends(
        taxonomy_df, utils.rejected_tags, "Sub Category", company_to_taxonomy_df, DR
    )
    .query("`Category` != 'Agritech'")
    .query("`Sub Category` != 'Alt protein (other)'")
)

magnitude_vs_growth_minor.sort_values("growth")

# %%
## Prep data for chart
# define sort order of points
major_sort_order = magnitude_vs_growth.sort_values("Growth").Category.to_list()
data = magnitude_vs_growth_minor.copy()
data["Category"] = pd.Categorical(data["Category"], categories=major_sort_order)
data = data.sort_values(["Category", "growth"], ascending=False)
data = data.assign(Increase=lambda df: df.growth > 0).assign(
    Magnitude=lambda df: df.Magnitude / 1e3
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
            axis=None,
        ),
        size=alt.Size(
            "Magnitude",
            title="Avg yearly investment (£ bn)",
            legend=alt.Legend(orient="left"),
            scale=alt.Scale(domain=[0.1, 4]),
        ),
        color=alt.Color(colour_field, legend=alt.Legend(orient="left")),
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
# Check companies
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
).head(1)

# %% [markdown]
# ### Subcategory time series

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
