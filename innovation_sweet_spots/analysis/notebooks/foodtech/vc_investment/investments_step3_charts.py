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
from innovation_sweet_spots.utils import chart_trends

output_folder = PROJECT_DIR / "outputs/foodtech/venture_capital"

import altair as alt
import pandas as pd
import numpy as np
import utils
import importlib
from collections import defaultdict
import itertools
from typing import DefaultDict

COLUMN_CATEGORIES = wu.dealroom.COLUMN_CATEGORIES

# %%
import innovation_sweet_spots.getters.dealroom
importlib.reload(innovation_sweet_spots.getters.dealroom);

# %%
# Initialise a Dealroom wrangler instance
DR = wu.DealroomWrangler()

# Check the number of companies
len(DR.company_data)

# %%
# Plotting utils
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver(path=alt_save.FIGURE_PATH + "/foodtech")

# Figure version name
VERSION_NAME = "Report_VC"

# %%
fig_filetypes = ["html", "svg", "png"]

# %% [markdown]
# ### Import reviewed data

# %%
# Check companies with rejected tags (relating to pets/animals)
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

# Fix these later upstream
minor_to_major['Retail'] = 'Restaurants and retail'
minor_to_major['Restaurants'] = 'Restaurants and retail'
company_to_taxonomy_df.loc[company_to_taxonomy_df["Category"] == 'Retail and restaurants', "Category"] = "Restaurants and retail" 
taxonomy_df.loc[taxonomy_df["Category"] == 'Retail and restaurants', "Category"] = "Restaurants and retail" 

minor_to_major['Supply chain'] = 'Delivery and logistics'
minor_to_major['Meal kits'] = 'Delivery and logistics'
minor_to_major['Delivery'] = 'Delivery and logistics'
company_to_taxonomy_df.loc[company_to_taxonomy_df["Category"] == 'Logistics', "Category"] = "Delivery and logistics" 
taxonomy_df.loc[taxonomy_df["Category"] == 'Logistics', "Category"] = "Delivery and logistics" 

minor_to_major['Alt protein (all)'] = 'Innovative food'

# %%
minor_to_major

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

# Adding the Alt protein (all) subcategory to the taxonomy dataframe
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
sorted(company_to_taxonomy_df.Category.unique())

# %%
kitchen_robot_ids = [
    4323337,
    4152603,
    3921341,
    3834707,
    3518716,
    3450868,
    3300579,
    3029048,
    2940904,
    2931036,
    2432763,
    1977744,
    1841335,
    1831995,
    1818775,
    1775048,
    1737517,
    1564994,
    1445841,
    1417916,
    1280760,
    966124,
    965511,
]
kitchen_robot_ids = [str(x) for x in kitchen_robot_ids]
df_kitchen_robots_cat = pd.DataFrame(
    data={
        "id": kitchen_robot_ids,
        "Category": "Cooking and kitchen",
        "level": "Category",
    }
)
df_kitchen_robots_subcat = pd.DataFrame(
    data={"id": kitchen_robot_ids, "Category": "Kitchen tech", "level": "Sub Category"}
)
company_to_taxonomy_df = pd.concat(
    [company_to_taxonomy_df, df_kitchen_robots_cat, df_kitchen_robots_subcat],
    ignore_index=True,
)

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
        min_year=2009,
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
        min_year=2009,
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
    alt.Tooltip(f"{values_column}:Q", format=",.3f", title=values_label),
]

data_early_late = foodtech_ts.assign(
    raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
)

fig = (
    alt.Chart(
        data_early_late.query("year > 2009"),
        width=400,
        height=200,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_column}:O", title=""),
        alt.Y(f"{values_column}:Q", title=values_label),
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
        data_early.query("year > 2009"),
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
    fig, f"v{VERSION_NAME}_total_early_investment", filetypes=fig_filetypes
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
i2011 = data_early.query("`year`==2011")[values_column].iloc[0]
i2021 = data_early.query("`year`==2021")[values_column].iloc[0]
print(i2011, i2021)
print('difference:', i2021 / i2011)

# %%
# Percentage difference between 2020 and 2021
au.percentage_change(
    data_early.query("`year`==2020")[values_column].iloc[0],
    data_early.query("`year`==2021")[values_column].iloc[0],
)

# %%
# Percentage difference between 2020 and 2021
au.percentage_change(
    data_early.query("`year`==2021")[values_column].iloc[0],
    data_early.query("`year`==2022")[values_column].iloc[0],
)

# %%
# Magnitude and growth between 2017 and 2021
au.estimate_magnitude_growth(data_early.drop(["deal_type"], axis=1), 2017, 2021)

# %%
# Magnitude and growth between 2017 and 2021
au.estimate_magnitude_growth(data_early.drop(["deal_type"], axis=1), 2011, 2021)

# %%
# data_early.to_csv('test.csv')

# %%
# Chart with only the late stage deals
data_late = data_early_late.query('deal_type == "Late"')

fig = (
    alt.Chart(
        data_late.query("year > 2009"),
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
        tooltip=tooltip,        
    )
    
)
fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig, f"v{VERSION_NAME}_total_late_investment", filetypes=fig_filetypes
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

data_early_late = foodtech_ts  # .query("year < 2022")

fig = (
    alt.Chart(
        data_early_late.query("year > 2009"),
        width=400,
        height=200,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_column}:O", title=""),
        alt.Y(f"{values_column}:Q", title=values_label),
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
    alt.Chart(df_major_amount_deal_type, width=350, height=300)
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
    # filetypes=["html", "svg", "png"],
    filetypes=fig_filetypes,
)

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
            title="Investment (proprtion)",
            axis=alt.Axis(labelFlush=False, format="%"),
            stack="normalize",
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

# %% [markdown]
# #### Logistics check

# %%
# Late stage deals time series
foodtech_logistics_ids = set(list(company_to_taxonomy_df.query("Category == 'Delivery and logistics'").id.unique()))
foodtech_logistics_ids = set(list(company_to_taxonomy_df.query("Category == 'Lab meat'").id.unique()))

foodtech_ts_late = (
    au.cb_get_all_timeseries(
        DR.company_data.query("id in @foodtech_logistics_ids"),
        (
            DR.funding_rounds.query("id in @foodtech_logistics_ids").query(
                "`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES"
            )
        ),
        period="year",
        min_year=2009,
        max_year=2022,
    )
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(deal_type="Late")
)


# %%
foodtech_ts_late

# %%
(612.536091-556.465087)/556.465087	

# %%
(3490.258340 - 1707.778906)/3490.258340


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
    taxonomy_df, utils.rejected_tags, company_to_taxonomy_df, DR, "Category"
)
# Get ids for each category
subcategory_ids = utils.get_category_ids(
    taxonomy_df, utils.rejected_tags, company_to_taxonomy_df, DR, "Sub Category"
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
print(f"Total foodtech funding is: £{round(funding_total / 1e3, 2)} Billion.")
print(
    f"Total foodtech funding (excl. Agritech) is: £{round(funding_total_minusAgritech / 1e3, 2)} Billion."
)

# %%
min_year = 2017
max_year = 2021

print(
    get_total_funding(category_ids["Delivery and logistics"], min_year=2017, max_year=2021)
    / funding_total
)
print(
    get_total_funding(category_ids["Delivery and logistics"], min_year=2017, max_year=2021)
    / funding_total_minusAgritech
)


def share_of_funding(
    min_year: int, max_year: int, category_ids: DefaultDict, category: str
):
    funding = get_total_funding(
        category_ids[category], min_year=min_year, max_year=max_year
    )
    print(
        f"The {category} category has received {round(funding / funding_total * 100, 2)}% of total foodtech funding."
    )
    print(
        f"The {category} category has received {round(funding / funding_total_minusAgritech * 100, 2)}% of total foodtech (excl. Agritech) funding."
    )


print(f"From {min_year} to {max_year}:")
share_of_funding(min_year, max_year, category_ids=category_ids, category="Health")
share_of_funding(min_year, max_year, category_ids=category_ids, category="Delivery and logistics")
share_of_funding(min_year, max_year, category_ids=subcategory_ids, category="Delivery")

# %% [markdown]
# ### Trends analysis

# %%
# Smoothed growth estimate of total venture funding 2017-2021, inferred from Crunchbase data
BASELINE_GROWTH = 1.28

# %%
import importlib
importlib.reload(utils);

# %%
# Caculate mangitude vs growth plots
magnitude_vs_growth = utils.get_trends(
    taxonomy_df, utils.rejected_tags, "Category", company_to_taxonomy_df, DR
)
magnitude_vs_growth_plot = magnitude_vs_growth.assign(
    magnitude=lambda df: df.Magnitude / 1e3
)

chart_trends.estimate_trend_type(magnitude_vs_growth).sort_values("growth")

# %%
importlib.reload(chart_trends);

# %%
# Chart configs
# mid point
mid_point = magnitude_vs_growth_plot.magnitude.median()
# color gradient width
chart_trends._epsilon = 0.05

fig = chart_trends.mangitude_vs_growth_chart(
    magnitude_vs_growth_plot,
    x_limit=18,
    y_limit=7,
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
    filetypes=["html"],
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
category = "Cooking and kitchen"
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
    # .query("year < 2022")
    .query("Category == @category")
)

fig = (
    alt.Chart(data, width=400, height=200)
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_column}:O", title=""),
        alt.Y(f"{values_column}:Q", title=values_label),
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
# ### Export major category trends results

# %%
TRENDS_DIR = PROJECT_DIR / f"outputs/foodtech/trends/"
TRENDS_DIR.mkdir(exist_ok=True, parents=True)
# Magnitude vs growth data
(
    chart_trends.estimate_trend_type(magnitude_vs_growth).to_csv(
        TRENDS_DIR / f"venture_capital_{VERSION_NAME}_Category.csv", index=False
    )
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

chart_trends.estimate_trend_type(magnitude_vs_growth_minor).sort_values("growth")

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
    alt.Chart(data, width=500, height=height)
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
            scale=alt.Scale(
                domain=(-1, 150),
                # domain=(.1, 100), type="log",                
            ),
        ),
        y=alt.Y("Sub Category:N", sort=data["Sub Category"].to_list(), axis=None),
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
    .encode(x=alt.X("x:Q"))
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
# ### Export minor subcategory trends results

# %%
(
    chart_trends.estimate_trend_type(magnitude_vs_growth_minor).to_csv(
        PROJECT_DIR
        / f"outputs/foodtech/trends/venture_capital_{VERSION_NAME}_SubCategory.csv",
        index=False,
    )
)

# %% [markdown]
# ### Subcategory time series

# %%
# Get company ids for each category
subcategory_ids = utils.get_category_ids(
    taxonomy_df, utils.rejected_tags, company_to_taxonomy_df, DR, "Sub Category"
)
# Get time series for each category
variable = "raised_amount_gbp_total"

subcategory_ts = (
    utils.get_category_ts(subcategory_ids, DR).rename(
        columns={"Category": "Sub Category"}
    )
    # .query("year < 2022")
)


# %%
subcategory_ts

# %%
importlib.reload(pu);

# %%
cats = ["Kitchen tech", "Dark kitchen"]

fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        subcategory_ts.assign(**{variable: lambda df: df[variable] / 1000}),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ billions)",
        category_column="Sub Category",
        amount_div=1,
    )
)
fig

# %%
# pu.configure_plots(pu.ts_bar_incomplete(
#     subcategory_ts,
#     categories_to_show=cats,
#     variable= "raised_amount_gbp_total",
#     variable_title= "Investment (£ billions)",
#     category_column= "Sub Category",
#     category_label= "Sub category",
#     amount_div= 1000,
#     width= 400,
#     height= 200,
#     tooltip=True,
# ))

# %%
AltairSaver.save(
    fig, f"v{VERSION_NAME}_ts_SubCategory_cooking", filetypes=["html", "svg", "png"]
)


# %%
cats = ["Lab meat", "Insects", "Plant-based", "Fermentation"]

fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        subcategory_ts.assign(**{variable: lambda df: df[variable] / 1000}),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ billions)",
        category_column="Sub Category",
        amount_div=1,
    )
)
fig

# %%
importlib.reload(pu);

# %%
pu.configure_plots(pu.ts_bar_incomplete(
    subcategory_ts,
    categories_to_show=cats,
    variable= "raised_amount_gbp_total",
    variable_title= "Investment (£ billions)",
    category_column= "Sub Category",
    category_label= "Sub-categories",
    amount_div= 1000,
    width= 450,
    height= 200,
    tooltip=True,
))

# %%
AltairSaver.save(
    fig, f"v{VERSION_NAME}_ts_SubCategory_alt_protein", filetypes=["html", "svg", "png"]
)


# %%
cats = ["Innovative food (other)", "Reformulation"]

fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        subcategory_ts.assign(**{variable: lambda df: df[variable] / 1000}),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ billions)",
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
    pu.ts_smooth_incomplete(
        subcategory_ts.assign(**{variable: lambda df: df[variable]}),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ millions)",
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
    pu.ts_smooth_incomplete(
        subcategory_ts.assign(**{variable: lambda df: df[variable]}),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ millions)",
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
    pu.ts_smooth_incomplete(
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

# %% [markdown]
# ## Exporting the list of startups

# %%
DR.company_data.head(1).columns[0:40]

# %%
pd.set_option("max_colwidth", 200)
ids = company_to_taxonomy_df
df_export = (
    company_to_taxonomy_df
    # Rename columns to avoid confusion
    .rename(columns={"Category": "category"})
    .query("level == 'Sub Category'")
    .drop_duplicates(["id", "category"])
    .merge(
        DR.company_data[
            [
                "id",
                "NAME",
                "TAGLINE",
                "TAGS",
                "INDUSTRIES",
                "SUB INDUSTRIES",
                "PROFILE URL",
                "WEBSITE",
                "country",
                "city",
                "LAUNCH DATE",
                "LAST FUNDING DATE",
            ]
        ],
        on="id",
        how="left",
    )
    # Fetch early deal values
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
    # add higher level category labels
    .merge(
        taxonomy_df[["Category", "Sub Category"]],
        left_on="category",
        right_on="Sub Category",
    )
    .drop(["funding_round_id", "raised_amount_usd", "level", "category"], axis=1)
    .rename(columns={"Sub Category": "sub_category", "Category": "category"})
    .sort_values(["raised_amount_gbp"], ascending=False)
    .sort_values(["category", "sub_category"])
)[
    [
        "id",
        "NAME",
        "TAGLINE",
        "PROFILE URL",
        "WEBSITE",
        "LAUNCH DATE",
        "country",
        "city",
        "raised_amount_gbp",
        "LAST FUNDING DATE",
        "TAGS",
        "INDUSTRIES",
        "SUB INDUSTRIES",
        "category",
        "sub_category",
    ]
]

df_export.loc[df_export.category == "agritech", "sub_category"] = "-"

# %%
df_export.head(1)

# %%
df_export.to_csv(
    PROJECT_DIR
    / "outputs/foodtech/venture_capital/foodtech_reviewed_VC_final_v2022_11_16.csv",
    index=False,
)

# %%
df_export.drop_duplicates(["id", "PROFILE URL"]).to_csv(
    PROJECT_DIR
    / "outputs/foodtech/venture_capital/foodtech_reviewed_VC_final_v2022_11_16_dedup.csv",
    index=False,
)

# %%
DR.company_data.query("NAME == 'Deliveroo'").to_csv(
    PROJECT_DIR / "outputs/foodtech/venture_capital/foodtech_reviewed_example_data.csv",
    index=False,
)

# %% [markdown]
# ## International comparison

# %% [markdown]
# ### Early deals

# %%
data_countries_early = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    # Add country data
    .merge(DR.company_data[["id", "country"]], how="left")
    .groupby(["country"], as_index=False)
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"))
    .assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1000)
    .sort_values("raised_amount_gbp", ascending=False)
    .drop_duplicates()
)
data_countries_early.head(10)

# %%
eu_early_funding = data_countries_early.query("country in @utils.EU_countries").sum()[
    "raised_amount_gbp"
]
print(
    f"In EU countries, there has been £{round(eu_early_funding, 2)} billion early investment from 2017 to 2022."
)

# %%
fig = (
    alt.Chart(data_countries_early.head(10), width=200, height=300)
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.Y(f"country:N", sort="-x", title=""),
        alt.X(
            f"raised_amount_gbp:Q",
            title="Investment (£ billions)"
            # scale=alt.Scale(domain=[0, 1200])
        ),
        tooltip=["country", "raised_amount_gbp"],
    )
)
fig = pu.configure_plots(fig, "", "")
fig

# %%
AltairSaver.save(fig, f"v{VERSION_NAME}_countries_early", filetypes=["html", "png"])

# %% [markdown]
# ### Late deals

# %%
data_countries_late = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.LATE_DEAL_TYPES")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    # Add country data
    .merge(DR.company_data[["id", "country"]])
    .groupby(["country"], as_index=False)
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"))
    .assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1000)
    .sort_values("raised_amount_gbp", ascending=False)
    .drop_duplicates()
)
data_countries_late.head(10)

# %%
eu_late_funding = data_countries_late.query("country in @utils.EU_countries").sum()[
    "raised_amount_gbp"
]
print(
    f"In EU countries, there has been £{round(eu_late_funding, 2)} billion late stage investment from 2017 to 2022."
)

# %%
fig = (
    alt.Chart(data_countries_late.head(10), width=200, height=300)
    .mark_bar(color=pu.NESTA_COLOURS[1])
    .encode(
        alt.Y(f"country:N", sort="-x", title=""),
        alt.X(f"raised_amount_gbp:Q", title="Investment (£ billions)"),
        tooltip=["country", "raised_amount_gbp"],
    )
)
fig = pu.configure_plots(fig, "", "")
fig

# %%
AltairSaver.save(fig, f"v{VERSION_NAME}_countries_late", filetypes=["html", "png"])

# %% [markdown]
# ### Early deals by category

# %%
# Dataframe with all countries and total early stage investments by category
data = (
    DR.funding_rounds.query("id in @foodtech_ids")
    .query("`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES")
    .query("announced_on >= '2017-01-01' and announced_on < '2022-01-01'")
    .merge(DR.company_data[["id", "country"]], how="left")
    .merge(
        company_to_taxonomy_df.query("level == 'Category'").drop_duplicates(
            ["id", "Category"]
        ),
        how="left",
    )
    .groupby(["country", "Category"], as_index=False)
    .agg(raised_amount_gbp=("raised_amount_gbp", "sum"))
    .assign(raised_amount_gbp=lambda df: df.raised_amount_gbp / 1000)
)

# Categories to show
cats = [
    "Delivery and logistics",
    "Innovative food",
    "Health",
    "Restaurants and retail",
    "Cooking and kitchen",
    "Food waste",
    # "Agritech",
]
top10_countries_per_cat = []
for cat in cats:
    top10_countries_per_cat.append(
        data.copy()
        .query("Category == @cat")
        .sort_values("raised_amount_gbp", ascending=False)
        .head(10)
    )
top10_countries_per_cat = pd.concat(top10_countries_per_cat, ignore_index=True)


# %%
data.query('Category == "Delivery and logistics"').raised_amount_gbp.sum()

# %%
# Plot top 10 countries for each category for early stage investments
fig = (
    alt.Chart((top10_countries_per_cat.query("Category in @cats")))
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        x=alt.X("raised_amount_gbp:Q", title="Investment (£ billions)"),
        y=alt.Y("country:N", sort="-x", title=""),
        facet=alt.Facet(
            "Category:N", title="", columns=2, header=alt.Header(labelFontSize=14)
        ),
        tooltip=[alt.Tooltip("raised_amount_gbp", format=".3f")],
    )
    .properties(width=180, height=180)
    .resolve_scale(x="independent", y="independent")
)


fig = pu.configure_plots(fig)
fig

# %%
AltairSaver.save(
    fig, f"v{VERSION_NAME}_countries_major_early", filetypes=["html", "png"]
)

# %% [markdown]
# ### Investment growth by countries

# %%
country = "United Kingdom"
countries = data_countries_early.country.head(10).to_list()

# %%
# Calculate early stage investment growth for countries
growth = []
for country in countries:
    df_companies = DR.company_data.query("id in @foodtech_ids").query(
        "country == @country"
    )
    df_companies_id = df_companies.id.to_list()

    country_ts_early = (
        au.cb_get_all_timeseries(
            df_companies,
            (
                DR.funding_rounds.query("id in @df_companies_id").query(
                    "`EACH ROUND TYPE` in @utils.EARLY_DEAL_TYPES"
                )
            ),
            period="year",
            min_year=2010,  # There are different values here
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
            2017,  # Compared to here
            2021,
        ).raised_amount_gbp_total
    )

countries_growth = pd.DataFrame({"country": countries, "growth": growth})


# %%
# View countries and their early stage investment growth
countries_growth.sort_values("growth")

# %%
# Add magnitude/raised amount data and divide growth by 100
countries_growth_magnitude = countries_growth.merge(
    data_countries_early, how="left"
).assign(growth=lambda df: df.growth / 100, magnitude=lambda df: df.raised_amount_gbp)
countries_growth_magnitude.sort_values("growth")

# %%
# Plot magnitude vs growth chart for countries' early stage investment

mid_point = countries_growth_magnitude.raised_amount_gbp.median()

fig = chart_trends.mangitude_vs_growth_chart(
    countries_growth_magnitude,
    x_limit=45,
    y_limit=20,
    mid_point=mid_point,
    baseline_growth=1.28,
    values_label="Investment (£ billions)",
    text_column="country",
    width=425,
)
fig.interactive()

# %%
# Plot countries' early stage investment growth
fig = (
    alt.Chart(countries_growth_magnitude, width=200, height=300)
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.Y(f"country:N", sort="-x", title=""),
        alt.X(f"growth:Q", title="Growth", axis=alt.Axis(format="%")),
    )
)
fig = pu.configure_plots(fig)
fig

# %% [markdown]
# ## Custom analyses

# %% [markdown]
# ### Custom combination of categories

# %%
trends_combined = (
    pd.concat([magnitude_vs_growth, magnitude_vs_growth_minor])
    .fillna("n/a (category level)")
    .sort_values(["Category", "Sub Category"])
    .reset_index(drop=True)
)
trends_combined.to_csv(
    PROJECT_DIR / f"outputs/foodtech/trends/venture_capital_{VERSION_NAME}_all.csv",
    index=False,
)

# %%
full_titles = [f'{x[0]}: {x[1]}' for x in list(zip(trends_combined['Category'], trends_combined['Sub Category']))]
trends_combined['full_titles'] = full_titles

# %%
categories_to_show = [
    'Innovative food: Alt protein (all)',
    'Delivery and logistics: Delivery',
    'Food waste: n/a (category level)',
    'Health: Biomedical',
    'Restaurants and retail: n/a (category level)',
    'Innovative food: Reformulation',
    'Delivery and logistics: Meal kits',
    'Cooking and kitchen: Dark kitchen',
    'Cooking and kitchen: Kitchen tech',
    'Health: Personalised nutrition',
]

# %%
trends_combined_ = (
    trends_combined.query('full_titles in @categories_to_show')
    .assign(magnitude=lambda df: df.Magnitude)
)

# %%
trends_combined_

# %%
import numpy as np

def log_growth(growth):
    if growth > 0:
        return np.log10(growth)
    else:
        return -np.log10(-growth)
    
    
trends_combined_log = trends_combined_.copy()
# trends_combined_log['magnitude'] = np.log10(trends_combined_log.magnitude)
# trends_combined_log['growth'] = trends_combined_log.Growth.apply(log_growth)

# %%
trends_combined_log

# %%
importlib.reload(alt);
importlib.reload(chart_trends);

# %%
trends_combined_log[['magnitude', 'growth', 'full_titles']]

# %%
trends_combined_log.magnitude.describe(percentiles=[0.25, 0.5, 0.75])

# %%
trends_combined_log.magnitude.median()

# %%
# importlib.reload(chart_trends);
mid_point = trends_combined_log.magnitude.median()
# color gradient width
chart_trends._epsilon = 0.01

fig = chart_trends.mangitude_vs_growth_chart(
    trends_combined_log,
    x_limit=20_000,
    y_limit=10,
    mid_point=mid_point,
    baseline_growth=BASELINE_GROWTH,
    values_label="Average investment per year (£ millions)",
    text_column='full_titles',
    width=425,
    horizontal_log=True,
    x_min=50,
)

# Baseline
baseline_rule = (
    alt.Chart(pd.DataFrame({"x": [mid_point]}))
    .mark_rule(strokeDash=[5, 7], size=1, color="k")
    .encode(
        x=alt.X(
            "x:Q",
        )
    )
)

baseline_rule_1= (
    alt.Chart(pd.DataFrame({"x": [192.682263]}))
    .mark_rule(strokeDash=[2, 2], size=.5, color="k")
    .encode(
        x=alt.X(
            "x:Q",
        )
    )
)

baseline_rule_2 = (
    alt.Chart(pd.DataFrame({"x": [1059.097782]}))
    .mark_rule(strokeDash=[2, 2], size=.5, color="k")
    .encode(
        x=alt.X(
            "x:Q",
        )
    )
)

fig = fig + baseline_rule + baseline_rule_1 + baseline_rule_2
fig

# fig = fig + baseline_rule
# fig

# %%
AltairSaver.save(
    fig, f"v{VERSION_NAME}_magnitude_growth_Synthesis_venture", filetypes=["html", "svg", "png"]
)

# %%
fig_points = (
    alt.Chart(trends_combined_log, width=350, height=350)
    .mark_circle(color="black")
    .encode(
        x=alt.X(
            "magnitude:Q",
            # title=horizontal_values_title,
            axis=alt.Axis(labelFlush=False,
                          labelExpr="pow(10, datum.value)"),
            scale=alt.Scale(domain=(1.6, 4.6)),
        ),
        y=alt.Y(
            "growth:Q",
            title="Growth",
            scale=alt.Scale(domain=(-2, 4.5)),            
            axis=alt.Axis(
                # format="%",
                labelFlush=False,
                labelExpr="datum.value > 0 ? pow(10, datum.value) + '%' : -pow(10, -datum.value) + '%'",
            ),

        ),
        tooltip=[
            alt.Tooltip(f"full_titles"),                
            # alt.Tooltip("magnitude:Q", title=horizontal_values_title, format=_values_format),
            # alt.Tooltip("growth:Q", title="Growth", format=".1%"),
        ],
    )
)
fig_points

# %%
import altair as alt
from vega_datasets import data

y2 = alt.value(0)

source = data.stocks()

alt.Chart(source.query("price > 100")).transform_filter(
    'datum.symbol==="GOOG"'
).mark_area(
    # line={'color':'darkgreen'},
    color=alt.Gradient(
        gradient='linear',
        stops=[alt.GradientStop(color='red', offset=0),
               alt.GradientStop(color='darkgreen', offset=1)],
        x1=1,
        x2=1,
        y1=0,
        y2=0,
    )
).encode(
    x=alt.X('date:T'),
    y=alt.Y('price:Q', scale=alt.Scale(type='linear', domain=(100, 1000))),
    y2 = alt.value(0),
)

# %% [markdown]
# ### Manually added

# %%
folder = "inputs/data/dealroom/raw_exports/foodtech_2022_11"
df = pd.read_excel(
    PROJECT_DIR / f"{folder}/df8381fc-6e7d-4eab-9ed5-59ecf66074dd.xlsx"
).astype({"id": str})

# %%
kitchen_robots_new = set(df.id.to_list()).difference(foodtech_ids)

# %%
df[df.id.isin(kitchen_robots_new)][["id", "NAME", "PROFILE URL", "WEBSITE"]].to_csv(
    PROJECT_DIR / f"{folder}/kitchen_robots_new_ids.csv"
)

# %% [markdown]
# ### Dealroom test sample

# %%
df = pd.read_excel(
    PROJECT_DIR / f"{folder}/Miguel_sample_1ebfe17f-3d74-46d4-b96b-b1e57f695088.xlsx"
).astype({"id": str})

# %%
df_random = df.sample(1)
ids = df_random.id.iloc[0]
df_random

# %%
DR.company_data.query("id == @ids")

# %%
DR2 = wu.DealroomWrangler(dataset="test")

# %%
DR2._company_data = DR2.process_input_data(df.copy())

# %%
DR2.funding_rounds.assign(year=lambda df: df.announced_on.dt.year).groupby(
    "year"
).count()

# %%
col = "EACH ROUND AMOUNT"
# col = 'EACH ROUND DATE'
print(df_random[col].iloc[0])
print(DR.company_data.query("id == @ids")[col].iloc[0])

# %%
set(df.id.to_list()).difference(DR.company_data.id.to_list())

# %%
# df.query("id == '3389'")

# %%
old_cols = set(DR.company_data.columns)
new_cols = set(df.columns)

# %%
# df['LAUNCH DATE']

# %%
# new_cols

# %%
old_cols.difference(new_cols)

# %%
# new_cols.difference(old_cols)

# %%
# # IDs obtained via manual checking
# kitchen_tech_robotics_ids = [
#     925686,
#     1653170,
#     1435371,
#     879531,
#     1763253,
#     948529,
#     988417,
#     988409,
#     1659462,
#     962941,
#     2031672,
#     174032,
#     1793126,
#     1659665,
#     1896957,
#     885072,
#     2019662,
#     1279421,
#     161884,
#     1398362,
#     1643637,
#     965184,
#     2016375,
#     232116,
#     134106,
#     1440458,
#     1818902,
#     1775243,
#     143736,
#     958578,
#     944819,
#     1784780,
#     1269910,
#     979885,
#     980802,
#     928421,
#     1818163,
# ]
# kitchen_tech_robotics_ids = [str(i) for i in kitchen_tech_robotics_ids]

# %%
category_ts = utils.get_category_ts(
    {"Kitchen robots": kitchen_tech_robotics_ids}, DR, deal_type=utils.EARLY_DEAL_TYPES
)
category_ts.head(1)

# %%
au.estimate_magnitude_growth(
    category_ts.drop(["Category", "time_period"], axis=1),
    year_start=2017,
    year_end=2021,
)

# %%
DR.company_data[DR.company_data.NAME.str.contains("Dexai")]

# %%
124.047030 * 5

# %%
pu.configure_plots(
    pu.ts_smooth(
        category_ts,
        ["Kitchen robots"],
        "raised_amount_gbp_total",
        "Investment (£ millions)",
        "Category",
        amount_div=1,
    )
)

# %% [markdown]
# ### Exporting funds

# %%
len(DR.company_data)

# %%
df = (
    DR.funding_rounds
    .merge(DR.company_data[['id', 'NAME', 'COMPANY STATUS', 'LAUNCH DATE', 'CLOSING DATE', 'TAGLINE', 'country', 'city', 'PROFILE URL', 'WEBSITE']], how='left')
    .merge(company_to_taxonomy_df[['id', 'Category', 'level']], how='left')
)

# %%
df.sort_values('announced_on')


# %%
def tag_deals(deal_type: str) -> str:
    if deal_type in utils.EARLY_DEAL_TYPES:
        return 'early'
    elif deal_type in utils.LATE_DEAL_TYPES:
        return 'late'
    else:
        return 'n/a'


# %%
df['deal_type'] = df['EACH ROUND TYPE'].apply(tag_deals)

# %%
df.to_csv(PROJECT_DIR / 'outputs/foodtech/interim/dealroom_foodtech_all_rounds.csv', index=False)

# %%
list(df.columns)

# %%
df[['NAME', 'Category', 'level']]

# %% [markdown]
# ### Investor stakeholders

# %%
investor_column = 'EACH ROUND INVESTORS'


# %%
def split_investors(text):
    if type(text) is str:
        return text.split("++")
    else:
        return []


# %%
df = (
    DR.funding_rounds
    .merge(DR.company_data[['id', 'NAME', 'COMPANY STATUS', 'LAUNCH DATE', 'CLOSING DATE', 'TAGLINE', 'country', 'city', 'PROFILE URL', 'WEBSITE']], how='left')
    .merge(company_to_taxonomy_df[['id', 'Category', 'level']], how='left')
    # .query("country == 'United Kingdom'")
)
df = df[-df[investor_column].isnull()]
df["investors"] = df[investor_column].apply(split_investors)
df = df.explode("investors")

# %%
# df.groupby(["investors", 

# %%
df_investors = (
    df.groupby(["level", "Category", "investors"], as_index=False)
    .agg(
        total_round_amount=("raised_amount_gbp", "sum"),
        companies=("NAME", lambda x: np.unique(x)),
    )
    .sort_values(["level", "Category", "total_round_amount"], ascending=False)
)

# %%
df_investors_unique = (
    df
    .drop_duplicates(["NAME", "investors"])
    .groupby(["investors"], as_index=False)
    .agg(
        total_round_amount=("raised_amount_gbp", "sum"),
        total_companies=("NAME", "count"),
        companies=("NAME", lambda x: np.unique(x)),
    )
    .sort_values(["total_companies"], ascending=False)
)

# %%
df_investors.to_csv(PROJECT_DIR / "outputs/foodtech/venture_capital/investors/investors_into_UK_startups.csv", index=False)

# %%
df_investors_unique.to_csv(PROJECT_DIR / "outputs/foodtech/venture_capital/investors/investors_into_startups_unique.csv", index=False)

# %%
