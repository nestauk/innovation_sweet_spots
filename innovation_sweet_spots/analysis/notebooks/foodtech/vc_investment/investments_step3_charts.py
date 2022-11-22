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

# Find companies with alt protein category
alt_protein_company_to_taxonomy_df = (
    company_to_taxonomy_df[company_to_taxonomy_df.Category.isin(alt_protein_cats)]
    .drop_duplicates("id")
    .copy()
)
# Set Category values to Alt protein (all)
alt_protein_company_to_taxonomy_df.loc[:, "Category"] = combined_category_name

# Combine original taxonomy with alt protein company taxonomy
company_to_taxonomy_df = pd.concat(
    [company_to_taxonomy_df, alt_protein_company_to_taxonomy_df], ignore_index=True
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
    alt.Chart(data_early_late, width=400, height=200)
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
    alt.Chart(data_early, width=400, height=200)
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_column}:O", title=""),
        alt.Y(f"{values_column}:Q", title=values_label),
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
    alt.Chart(data_late, width=400, height=200)
    .mark_bar(color=pu.NESTA_COLOURS[1])
    .encode(
        alt.X(f"{horizontal_column}:O", title=""),
        alt.Y(f"{values_column}:Q", title=values_label),
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
    alt.Chart(data_early_late, width=400, height=200)
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
    filetypes=["html", "svg", "png"],
)

# %% [markdown]
# ### Category proportion of total investment

# %%
DR.funding_rounds


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
    taxonomy_df,
    utils.rejected_tags,
    company_to_taxonomy_df,
    DR,
    "Sub Category",
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
health_funding = get_total_funding(
    category_ids["Health"], min_year=min_year, max_year=max_year
)
print(f"From {min_year} to {max_year}:")
print(
    f"The Health category has received {round(health_funding / funding_total * 100, 2)}% of total foodtech funding."
)
print(
    f"The Health category has received {round(health_funding / funding_total_minusAgritech * 100, 2)}% of total foodtech (excl. Agritech) funding."
)

logistics_funding = get_total_funding(
    category_ids["Logistics"], min_year=min_year, max_year=max_year
)
print(
    f"The Logistics category has received {round(logistics_funding / funding_total * 100, 2)}% of total foodtech funding."
)
print(
    f"The Logistics category has received {round(logistics_funding / funding_total_minusAgritech * 100, 2)}% of total foodtech (excl. Agritech) funding."
)

# %%
print(
    get_total_funding(subcategory_ids["Delivery"], min_year=2017, max_year=2021)
    / funding_total
)
print(
    get_total_funding(subcategory_ids["Delivery"], min_year=2017, max_year=2021)
    / funding_total_minusAgritech
)

# %% [markdown]
# ### Trends analysis

# %%
# Smoothed growth estimate of total venture funding 2017-2021, inferred from Crunchbase data
BASELINE_GROWTH = 1.28

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
# Chart configs
# mid point
mid_point = magnitude_vs_growth_plot.magnitude.median()
# color gradient width
chart_trends._epsilon = 0.05

fig = chart_trends.mangitude_vs_growth_chart(
    magnitude_vs_growth_plot,
    x_limit=16,
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
            scale=alt.Scale(domain=(-1, 100)),
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
    utils.get_category_ts(subcategory_ids, DR)
    .rename(columns={"Category": "Sub Category"})
    .query("year < 2022")
)


# %%
subcategory_ts

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
                "PROFILE URL",
                "WEBSITE",
                "country",
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
        "raised_amount_gbp",
        "LAST FUNDING DATE",
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
    "Logistics",
    "Innovative food",
    "Health",
    "Retail and restaurants",
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
data.query('Category == "Logistics"').raised_amount_gbp.sum()

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
# ## Custom combination of categories

# %%
trends_combined = (
    pd.concat([magnitude_vs_growth, magnitude_vs_growth_minor])
    .fillna("n/a (category level)")
    .sort_values(["Category", "Sub Category"])
)
trends_combined.to_csv(
    PROJECT_DIR / f"outputs/foodtech/trends/venture_capital_{VERSION_NAME}_all.csv",
    index=False,
)

# %%
chart_trends.estimate_trend_type(trends_combined)

# %%
