# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: innovation_sweet_spots
#     language: python
#     name: python3
# ---

# # Descriptive analysis
#
# - Descriptive analysis (number of companies per category etc)
# - Trends per category
# - Visualise trends
#
# - Discuss improvements to the taxonomy and categorisation
# - Approach to validation

# # Setup

from innovation_sweet_spots import PROJECT_DIR
import innovation_sweet_spots.utils.google_sheets as gs
from innovation_sweet_spots.analysis import wrangling_utils as wu
import re
import utils
import innovation_sweet_spots.utils.plotting_utils as pu
import pandas as pd
import altair as alt
import innovation_sweet_spots.analysis.analysis_utils as au
import importlib

# +
FIGURES_DIR = PROJECT_DIR / "outputs/2023_childcare/figures/"
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver(path=FIGURES_DIR)
TABLES_DIR = FIGURES_DIR / "tables"
filetypes = ["html"]

OUTPUTS_TABLE = "107PT9NFeTrIUVhgMKwu-3PlAMhpMg0pH3wPdb0b9pAQ"
# -

# ## Helper functions


# +
def select_by_role(cb_orgs: pd.DataFrame, role: str):
    """
    Select companies that have the specified role.
    Roles can be 'investor', 'company', or 'both'
    """
    all_roles = cb_orgs.roles.copy().fillna("")
    if role != "both":
        return cb_orgs[all_roles.str.contains(role)]
    else:
        return cb_orgs[
            all_roles.str.contains("investor") & all_roles.str.contains("company")
        ]


def save_data_table(table: pd.DataFrame, filename: str, folder):
    """Helper function to save table data underpinning figures"""
    table.to_csv(folder / f"{filename}.csv", index=False)


def percentage_change(
    ts_df: pd.DataFrame, first_year: int, second_year: int, v: bool = False
) -> float:
    p = au.percentage_change(
        ts_df.query("year == @first_year").raised_amount_gbp_total.iloc[0],
        ts_df.query("year == @second_year").raised_amount_gbp_total.iloc[0],
    )
    if v:
        # Print and round to 3 decimal places
        print(f"Percentage change from {first_year} to {second_year}: {p:.3f}")
    else:
        return p


# -

# ## Loading in the data

CB = wu.CrunchbaseWrangler()

# +
# childcare_categories_df = pd.read_csv(
#     PROJECT_DIR / "outputs/2023_childcare/finals/company_to_subtheme_v2023_05_16.csv"
# ).drop_duplicates(["cb_id", "subtheme_tag"])

childcare_categories_df = pd.read_csv(
    PROJECT_DIR / "outputs/2023_childcare/finals/company_to_subtheme_v2023_06_06.csv"
).drop_duplicates(["cb_id", "subtheme_tag"])

# +
# Company identifiers
companies_ids = set(childcare_categories_df.cb_id.to_list())
companies_df = (
    CB.cb_organisations.query("id in @companies_ids")
    .query("country in @utils.list_of_select_countries")
    .pipe(select_by_role, "company")
)
childcare_categories_df = childcare_categories_df.query(
    "cb_id in @companies_df.id.to_list()"
)

# Funding data
funding_df = CB.get_funding_rounds(companies_df).query(
    "investment_type in @utils.EARLY_STAGE_DEALS or investment_type in @utils.LATE_STAGE_DEALS"
)

funding_only_early_df = CB.get_funding_rounds(companies_df).query(
    "investment_type in @utils.EARLY_STAGE_DEALS"
)
# -

# check double counting
print(funding_df.duplicated(["org_id", "funding_round_id"], keep=False).sum())
print(
    funding_only_early_df.duplicated(["org_id", "funding_round_id"], keep=False).sum()
)

# # Analysis

# ## Composition of companies by themes

# Total number of companies
len(companies_df.id.to_list())

companies_df.id.duplicated().sum()

theme_counts_df = (
    childcare_categories_df.query("cb_id in @companies_df.id.to_list()")
    .drop_duplicates(["cb_id", "theme"])
    .groupby(["theme"])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
    .sort_values(["count"], ascending=False)
    .reset_index(drop=True)
)

theme_counts_df

name = "theme_counts"
save_data_table(theme_counts_df, name, TABLES_DIR)
gs.upload_to_google_sheet(theme_counts_df, OUTPUTS_TABLE, name, overwrite=True)


sort_order = theme_counts_df.theme.to_list()
subtheme_counts_df = (
    childcare_categories_df.drop_duplicates(["cb_id", "subtheme_full"])
    .groupby(["theme", "subtheme_full"])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
    # make theme categorical and provide sort_order list as the sorting order
    .astype({"theme": "category"})
    .assign(theme=lambda x: x.theme.cat.set_categories(sort_order))
    .sort_values(["theme", "count"], ascending=[True, False])
)
subtheme_counts_df

name = "subtheme_counts"
save_data_table(subtheme_counts_df, name, TABLES_DIR)
gs.upload_to_google_sheet(subtheme_counts_df, OUTPUTS_TABLE, name, overwrite=True)

# ## Establishing baselines

# ### Global baseline
#
# Including all companies in the scope

# +
# Company identifiers
baseline_companies_df = (
    CB.cb_organisations.query("country in @utils.list_of_select_countries")
    .query("founded_on > '1900-01-01'")
    .pipe(select_by_role, "company")
)

# Funding data
baseline_funding_df = CB.get_funding_rounds(baseline_companies_df).query(
    "investment_type in @utils.EARLY_STAGE_DEALS or investment_type in @utils.LATE_STAGE_DEALS"
)

baseline_funding_only_early_df = CB.get_funding_rounds(baseline_companies_df).query(
    "investment_type in @utils.EARLY_STAGE_DEALS"
)
# -

# check double counting
print(baseline_funding_df.duplicated(["org_id", "funding_round_id"], keep=False).sum())
print(
    baseline_funding_only_early_df.duplicated(
        ["org_id", "funding_round_id"], keep=False
    ).sum()
)

# +
# Funding time series
baseline_funding_ts = au.cb_get_all_timeseries(
    baseline_companies_df, baseline_funding_df, "year", 2009, 2023
).assign(year=lambda df: df.time_period.dt.year)

# Funding time series
baseline_funding_only_early_ts = au.cb_get_all_timeseries(
    baseline_companies_df, baseline_funding_only_early_df, "year", 2009, 2023
).assign(year=lambda df: df.time_period.dt.year)
# -

baseline_funding_only_early_ts

name = "baseline_total_early_funding_ts"
save_data_table(baseline_funding_only_early_ts, name, TABLES_DIR)
gs.upload_to_google_sheet(
    baseline_funding_only_early_ts, OUTPUTS_TABLE, name, overwrite=True
)

percentage_change(baseline_funding_only_early_ts, 2011, 2021, v=True)
percentage_change(baseline_funding_only_early_ts, 2020, 2021, v=True)
percentage_change(baseline_funding_only_early_ts, 2021, 2022, v=True)

au.ts_magnitude_growth_(baseline_funding_only_early_ts, 2018, 2022)

# ### Edtech baseline

"e-learning" in CB.industries

# +
edtech_companies_df = (
    CB.get_companies_in_industries(["edtech", "education", "e-learning"])
    .query("founded_on > '1900-01-01'")
    .query("country in @utils.list_of_select_countries")
    .pipe(select_by_role, "company")
)

# Funding data
edtech_funding_df = CB.get_funding_rounds(edtech_companies_df).query(
    "investment_type in @utils.EARLY_STAGE_DEALS"
)

# Funding time series
edtech_funding_ts = (
    au.cb_get_all_timeseries(edtech_companies_df, edtech_funding_df, "year", 2012, 2023)
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1e3)
)
# -

len(edtech_companies_df)

edtech_funding_ts

3426.632853 / 689.952021

1647.744888 / 338.137485

percentage_change(edtech_funding_ts, 2012, 2021, v=True)
percentage_change(edtech_funding_ts, 2020, 2021, v=True)
percentage_change(edtech_funding_ts, 2021, 2022, v=True)

percentage_change(edtech_funding_ts, 2012, 2022, v=True)

# ### UK baseline

# +
# # Baseline
# baseline_funding_df_uk = (
#     baseline_funding_only_early_df
#     .merge(companies_df[["id", "country"]], left_on="org_id", right_on="id")
#     .query("year >= 2012 and year <= 2023")
#     .copy()
# )

all_companies_df_uk = (
    CB.cb_organisations.query("country == 'United Kingdom'")
    .query("founded_on > '1900-01-01'")
    .copy()
)

# Funding data
baseline_funding_df_uk = CB.get_funding_rounds(all_companies_df_uk).query(
    "investment_type in @utils.EARLY_STAGE_DEALS"
)


# baseline_funding_df_uk = (
#     baseline_funding_df_uk
#     .query("org_id in @all_companies_df_uk.id.to_list()")
# )

# Funding time series
baseline_funding_ts_uk = (
    au.cb_get_all_timeseries(
        all_companies_df_uk, baseline_funding_df_uk, "year", 2012, 2023
    )
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1e3)
)

# +
# Funding data
baseline_funding_late_df_uk = CB.get_funding_rounds(all_companies_df_uk).query(
    "investment_type in @utils.EARLY_STAGE_DEALS or investment_type in @utils.LATE_STAGE_DEALS"
)

# Funding time series
baseline_funding_late_ts_uk = (
    au.cb_get_all_timeseries(
        all_companies_df_uk, baseline_funding_late_df_uk, "year", 2012, 2023
    )
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1e3)
)
# -

# check double counting
print(
    baseline_funding_df_uk.duplicated(["org_id", "funding_round_id"], keep=False).sum()
)

baseline_funding_ts_uk

percentage_change(baseline_funding_ts_uk, 2020, 2021, v=True)
percentage_change(baseline_funding_ts_uk, 2021, 2022, v=True)

# ## Company funding

# Types of allowed deals
utils.EARLY_STAGE_DEALS

utils.LATE_STAGE_DEALS

# +
deal_order = ["n/a", "£0-5M", "£5-20M", "£20-100M", "£100M+"]

funding_df_ranges = (
    funding_df.assign(
        deal_type=lambda df: df.raised_amount_gbp.apply(
            utils.deal_amount_to_range_coarse
        )
    )
    .astype({"deal_type": "category"})
    .assign(deal_type=lambda x: x.deal_type.cat.set_categories(deal_order))
    .assign(year=lambda df: pd.to_datetime(df.announced_on).dt.year)
)

funding_only_early_df_ranges = (
    funding_only_early_df.assign(
        deal_type=lambda df: df.raised_amount_gbp.apply(
            utils.deal_amount_to_range_coarse
        )
    )
    .astype({"deal_type": "category"})
    .assign(deal_type=lambda x: x.deal_type.cat.set_categories(deal_order))
    .assign(year=lambda df: pd.to_datetime(df.announced_on).dt.year)
)

# +
# Total funding per company
companies_funding_df = funding_only_early_df.groupby("org_id", as_index=False).agg(
    total_funding_gbp=("raised_amount_gbp", "sum")
)

# Funding per company and theme
companies_funding_theme_df = (
    companies_df.merge(companies_funding_df, left_on="id", right_on="org_id")
    .merge(childcare_categories_df, left_on="id", right_on="cb_id")
    .drop(["org_id", "cb_id"], axis=1)
)

# Export a prelim funding table per company
columns_to_export = [
    "id",
    "name",
    "cb_url",
    "short_description",
    "country",
    "total_funding_gbp",
    "theme",
    "subtheme",
    "subtheme_full",
    "subtheme_tag",
]
companies_funding_theme_df[columns_to_export].to_csv(
    TABLES_DIR / "companies_funding_theme.csv", index=False
)
# -

# ### Total investment over time

# +
# Funding time series
funding_ts = au.cb_get_all_timeseries(
    companies_df, funding_df, "year", 2009, 2023
).assign(year=lambda df: df.time_period.dt.year)

funding_only_early_ts = au.cb_get_all_timeseries(
    companies_df, funding_only_early_df, "year", 2009, 2023
).assign(year=lambda df: df.time_period.dt.year)
# -

funding_only_early_ts

percentage_change(funding_only_early_ts, 2011, 2021, v=True)
percentage_change(funding_only_early_ts, 2020, 2021, v=True)
percentage_change(funding_only_early_ts, 2021, 2022, v=True)

name = "total_funding_early_ts"
save_data_table(funding_only_early_ts, name, TABLES_DIR)
gs.upload_to_google_sheet(funding_only_early_ts, OUTPUTS_TABLE, name, overwrite=True)

# ### Deal size time series

# +
# altair stacked bar chart with number of deals, with deal types as color
data = (
    funding_only_early_df_ranges.groupby(["year", "deal_type"], as_index=True)
    .agg(
        counts=("funding_round_id", "count"),
        total_amount=("raised_amount_gbp", "sum"),
    )
    .reset_index()
    .query("year > 2009")
    .query("year <= 2023")
    # convert to millions
    .assign(total_amount=lambda df: df.total_amount / 1000)
)

# turn the datafrom from long to wide
data_wide_df = (
    data.pivot(index="year", columns="deal_type", values="total_amount")
    .fillna(0)
    .astype(int)
    .reset_index()
)

data_wide_df
# -


name = "funding_ts_deal_ranges"
save_data_table(data_wide_df, name, TABLES_DIR)
gs.upload_to_google_sheet(data_wide_df, OUTPUTS_TABLE, name, overwrite=True)

fig = (
    alt.Chart(
        data,
        width=400,
        height=300,
    )
    .mark_bar()
    .encode(
        alt.X("year:O", title=""),
        alt.Y("counts:Q", title="Number of deals"),
        alt.Color("deal_type:N", title="Deal size", sort=deal_order),
        tooltip=["year", "counts", "deal_type"],
    )
)
pu.configure_plots(fig)

fig = (
    alt.Chart(
        data,
        width=400,
        height=300,
    )
    .mark_bar()
    .encode(
        alt.X("year:O", title=""),
        alt.Y("total_amount:Q", title="Investment (£ millions)"),
        alt.Color("deal_type:N", title="Deal size", sort=deal_order),
        tooltip=["year", "counts", "deal_type"],
    )
)
pu.configure_plots(fig)

# ## Company funding by theme

# ### Time series

# +
themes_list = childcare_categories_df.theme.unique()
subthemes_list = childcare_categories_df.subtheme_full.unique()
themes_to_subthemes = utils.get_taxonomy_dict(
    childcare_categories_df, "theme", "subtheme_full"
)

themes_to_ids = {
    theme: set(childcare_categories_df.query("theme == @theme").cb_id)
    for theme in themes_list
}
subthemes_to_ids = {
    subtheme: set(childcare_categories_df.query("subtheme_full == @subtheme").cb_id)
    for subtheme in subthemes_list
}
# -

# Collect all theme time series (NB: Double counting will occur, so don't sum them)
themes_ts = []
for theme in themes_list:
    theme_ids = themes_to_ids[theme]
    # Get companies in a theme
    companies_theme_df = companies_df.query("id in @theme_ids").copy()
    assert companies_theme_df.duplicated("id").sum() == 0
    # Get funding for companies in a theme
    funding_theme_df = funding_only_early_df.query("org_id in @theme_ids").copy()
    test_df = CB.get_funding_rounds(companies_theme_df).query(
        "investment_type in @utils.EARLY_STAGE_DEALS"
    )
    assert len(funding_theme_df) == len(test_df)
    # Funding time series
    funding_ts = au.cb_get_all_timeseries(
        companies_theme_df, funding_theme_df, "year", 2012, 2023
    ).assign(theme=theme)
    # Collect all time series
    themes_ts.append(funding_ts)
# Final dataframe
themes_ts = pd.concat(themes_ts, ignore_index=True).assign(
    year=lambda df: df.time_period.dt.year
)

# turn the datafrom from long to wide
themes_wide_ts = (
    themes_ts.pivot(index="year", columns="theme", values="raised_amount_gbp_total")
    .fillna(0)
    .astype(int)
    .reset_index()
)


name = "themes_ts"
save_data_table(themes_wide_ts, name, TABLES_DIR)
gs.upload_to_google_sheet(themes_wide_ts, OUTPUTS_TABLE, name, overwrite=True)

# Collect all subtheme time series (NB: Double counting will occur, so don't sum them)
subthemes_ts = []
for subtheme in subthemes_list:
    subtheme_ids = subthemes_to_ids[subtheme]
    # Get companies in a theme
    companies_subtheme_df = companies_df.query("id in @subtheme_ids").copy()
    assert companies_theme_df.duplicated("id").sum() == 0
    # Get funding for companies in a theme
    funding_subtheme_df = funding_only_early_df.query("org_id in @subtheme_ids").copy()
    test_df = CB.get_funding_rounds(companies_subtheme_df).query(
        "investment_type in @utils.EARLY_STAGE_DEALS"
    )
    assert len(funding_subtheme_df) == len(test_df)
    # Funding time series
    funding_ts = au.cb_get_all_timeseries(
        companies_subtheme_df, funding_subtheme_df, "year", 2010, 2022
    ).assign(subtheme=subtheme)
    # Collect all time series
    subthemes_ts.append(funding_ts)
# Final dataframe
subthemes_ts = pd.concat(subthemes_ts, ignore_index=True).assign(
    year=lambda df: df.time_period.dt.year
)

# turn the datafrom from long to wide
subthemes_wide_ts = (
    subthemes_ts.pivot(
        index="year", columns="subtheme", values="raised_amount_gbp_total"
    )
    .fillna(0)
    .astype(int)
    .reset_index()
)


subthemes_wide_ts

name = "subthemes_ts"
save_data_table(subthemes_wide_ts, name, TABLES_DIR)
gs.upload_to_google_sheet(subthemes_wide_ts, OUTPUTS_TABLE, name, overwrite=True)

# ### Total funding by theme

# +
# Total funding per company
companies_funding_period_df = (
    funding_only_early_df.query("year >= 2018 and year <= 2022")
    .groupby("org_id", as_index=False)
    .agg(total_funding_gbp=("raised_amount_gbp", "sum"))
)


companies_funding_theme_period_df = (
    companies_df.merge(companies_funding_period_df, left_on="id", right_on="org_id")
    .merge(childcare_categories_df, left_on="id", right_on="cb_id")
    .drop(["org_id", "cb_id"], axis=1)
)

# +
theme_total_funding_df = (
    companies_funding_theme_period_df.groupby(["theme"], as_index=False)
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .sort_values("total_funding_gbp", ascending=False)
)

theme_total_funding_df

# +
sort_order = theme_total_funding_df.theme.to_list()

subtheme_total_funding_df = (
    companies_funding_theme_period_df.groupby(
        ["theme", "subtheme_full"], as_index=False
    )
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .sort_values(["theme", "total_funding_gbp"], ascending=[True, False])
)

subtheme_total_funding_df
# -

# ### Funding by theme and deal size

# +
# Total funding per company
companies_funding_period_df = (
    funding_only_early_df_ranges.query("year >= 2018 and year <= 2022")
    .groupby(["org_id", "deal_type"])
    .agg(total_funding_gbp=("raised_amount_gbp", "sum"))
    .reset_index()
)

companies_funding_theme_period_df = (
    companies_df.merge(companies_funding_period_df, left_on="id", right_on="org_id")
    .merge(childcare_categories_df, left_on="id", right_on="cb_id")
    .drop(["org_id", "cb_id"], axis=1)
)

theme_funding_df = (
    companies_funding_theme_period_df.drop_duplicates(
        subset=["theme", "id", "deal_type"]
    )
    .groupby(["theme", "deal_type"], as_index=False)
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .query("deal_type != 'n/a'")
    .sort_values(["theme", "deal_type"], ascending=[True, True])
)

# turn the datafrom from long to wide
theme_funding_wide_df = (
    theme_funding_df.pivot(
        index="theme", columns="deal_type", values="total_funding_gbp"
    )
    .fillna(0)
    .astype(int)
    .reset_index()
    .merge(theme_total_funding_df, on="theme", how="left")
    .sort_values("total_funding_gbp", ascending=False)
)

theme_funding_wide_df
# -

name = "funding_themes"
save_data_table(theme_funding_wide_df, name, TABLES_DIR)
gs.upload_to_google_sheet(theme_funding_wide_df, OUTPUTS_TABLE, name, overwrite=True)

# +
subtheme_funding_df = (
    companies_funding_theme_period_df.drop_duplicates(
        subset=["subtheme_full", "id", "deal_type"]
    )
    .groupby(["subtheme_full", "deal_type"], as_index=True)
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .query("deal_type != 'n/a'")
    .reset_index()
)

# #turn the datafrom from long to wide
subtheme_funding_wide_df = (
    subtheme_funding_df.pivot(
        index="subtheme_full", columns="deal_type", values="total_funding_gbp"
    )
    .fillna(0)
    .astype(int)
    .merge(subtheme_total_funding_df, on="subtheme_full", how="left")
    .sort_values("total_funding_gbp", ascending=False)
)

subtheme_funding_wide_df
# -

name = "funding_subthemes"
save_data_table(subtheme_funding_wide_df, name, TABLES_DIR)
gs.upload_to_google_sheet(subtheme_funding_wide_df, OUTPUTS_TABLE, name, overwrite=True)

# ### Funding growth by theme

dfs = []
for theme in themes_ts.theme.unique():
    df = (
        themes_ts.query("theme == @theme")
        .sort_values("year")
        .drop(["theme", "time_period"], axis=1)
    ).copy()
    dfs.append(
        au.ts_magnitude_growth_(df, 2018, 2022)
        .assign(theme=theme)
        .reset_index()
        .rename(columns={"index": "variable"})
    )
dfs = pd.concat(dfs, ignore_index=True)
dfs.head(5)

# +
theme_growth_df = dfs.query("variable == 'raised_amount_gbp_total'").sort_values(
    "growth", ascending=False
)

name = "growth_theme"
save_data_table(theme_growth_df, name, TABLES_DIR)
gs.upload_to_google_sheet(theme_growth_df, OUTPUTS_TABLE, name, overwrite=True)
# -

dfs = []
for subtheme in subthemes_ts.subtheme.unique():
    df = (
        subthemes_ts.query("subtheme == @subtheme")
        .sort_values("year")
        .drop(["subtheme", "time_period"], axis=1)
    ).copy()
    dfs.append(
        au.ts_magnitude_growth_(df, 2018, 2022)
        .assign(subtheme=subtheme)
        .reset_index()
        .rename(columns={"index": "variable"})
    )
dfs = pd.concat(dfs, ignore_index=True)
dfs.head(5)

# +
subtheme_growth_df = dfs.query("variable == 'raised_amount_gbp_total'").sort_values(
    "growth", ascending=False
)

name = "growth_subtheme"
save_data_table(subtheme_growth_df, name, TABLES_DIR)
gs.upload_to_google_sheet(subtheme_growth_df, OUTPUTS_TABLE, name, overwrite=True)
# -

# ## Country comparison

# ### Baseline country funding

# +
# Total funding per company
baseline_companies_funding_period_df = (
    baseline_funding_only_early_df.query("year >= 2018 and year <= 2022")
    .groupby(["org_id"], as_index=True)
    .agg(
        total_funding_gbp=("raised_amount_gbp", "sum"),
        deal_count=("funding_round_id", "count"),
    )
    .reset_index(drop=False)
)

baseline_companies_funding_country_period_df = (
    CB.cb_organisations[["id", "country"]]
    .query("country in @utils.list_of_select_countries")
    .merge(
        baseline_companies_funding_period_df,
        left_on="id",
        right_on="org_id",
        how="left",
    )
    .drop(["org_id"], axis=1)
)

baseline_country_funding_df = (
    baseline_companies_funding_country_period_df.drop_duplicates(
        subset=["country", "id"]
    )
    .groupby(["country"], as_index=False)
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .sort_values(["total_funding_gbp"], ascending=False)
)

# +
df = baseline_country_funding_df
us = df[df.country == "United States"].total_funding_gbp.iloc[0]
uk = df[df.country == "United Kingdom"].total_funding_gbp.iloc[0]
eu = df[df.country.isin(utils.list_of_countries_in_europe())].sum().total_funding_gbp

print(f"Proportion of UK vs US funding: {uk/us:.2f}")
print(f"Proportion of Europe vs US funding: {eu/us:.2f}")
# -

# ### Funding by country

# +
# Total funding per company
companies_funding_period_df = (
    funding_only_early_df_ranges.query("year >= 2018 and year <= 2022")
    .groupby(["org_id", "deal_type"], as_index=True)
    .agg(
        total_funding_gbp=("raised_amount_gbp", "sum"),
        deal_count=("funding_round_id", "count"),
    )
    .reset_index(drop=False)
)

companies_funding_country_period_df = (
    companies_df[["id", "country"]]
    .merge(companies_funding_period_df, left_on="id", right_on="org_id")
    .drop(["org_id"], axis=1)
)

country_funding_df = (
    companies_funding_country_period_df.drop_duplicates(
        subset=["country", "id", "deal_type"]
    )
    .groupby(["country", "deal_type"], as_index=False)
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .sort_values(["deal_type", "total_funding_gbp"], ascending=False)
    .query("deal_type != 'n/a'")
    .sort_values(["country", "deal_type"], ascending=[True, True])
)

total_funding_df = (
    country_funding_df.groupby("country")
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .sort_values("total_funding_gbp", ascending=False)
)

# turn the datafrom from long to wide
country_funding_wide_df = (
    country_funding_df.pivot(
        index="country", columns="deal_type", values="total_funding_gbp"
    )
    .fillna(0)
    .astype(int)
    .reset_index()
    .merge(total_funding_df, on="country", how="left")
    .sort_values("total_funding_gbp", ascending=False)
    .head(13)
    .reset_index(drop=True)
)

country_funding_wide_df

# +
df = total_funding_df.reset_index()
us = df[df.country == "United States"].total_funding_gbp.iloc[0]
uk = df[df.country == "United Kingdom"].total_funding_gbp.iloc[0]
eu = df[df.country.isin(utils.list_of_countries_in_europe())].sum().total_funding_gbp

print(f"Proportion of UK vs US funding: {uk/us:.2f}")
print(f"Proportion of Europe vs US funding: {eu/us:.2f}")
# -

name = "funding_countries"
save_data_table(country_funding_wide_df, name, TABLES_DIR)
gs.upload_to_google_sheet(country_funding_wide_df, OUTPUTS_TABLE, name, overwrite=True)

# ### Baseline edtech funding by country

# +
# Total funding per company
edtech_companies_funding_period_df = (
    edtech_funding_df.query("year >= 2018 and year <= 2022")
    .groupby(["org_id"], as_index=True)
    .agg(
        total_funding_gbp=("raised_amount_gbp", "sum"),
        deal_count=("funding_round_id", "count"),
    )
    .reset_index(drop=False)
)

edtech_companies_funding_country_period_df = (
    edtech_companies_df[["id", "country"]]
    .merge(edtech_companies_funding_period_df, left_on="id", right_on="org_id")
    .drop(["org_id"], axis=1)
)

edtech_country_funding_df = (
    edtech_companies_funding_country_period_df.drop_duplicates(subset=["country", "id"])
    .groupby(["country"], as_index=False)
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .sort_values("total_funding_gbp", ascending=False)
    .assign(total_funding_gbp=lambda x: round(x.total_funding_gbp / 1e3, 3))
    .reset_index(drop=True)
)

# edtech_total_funding_df = (
#     edtech_country_funding_df
#     .groupby('country')
#     .agg(total_funding_gbp=('total_funding_gbp', 'sum'))
#     .sort_values('total_funding_gbp', ascending=False)
# )

edtech_country_funding_df
# -

# ### UK investment

# +
funding_df_uk = (
    funding_only_early_df.merge(
        companies_df[["id", "country"]], left_on="org_id", right_on="id"
    )
    .query("year >= 2012 and year <= 2023")
    .query("country == 'United Kingdom'")
    .copy()
)
companies_df_uk = companies_df.query("country == 'United Kingdom'").copy()

# Funding time series
funding_ts_uk = (
    au.cb_get_all_timeseries(companies_df_uk, funding_df_uk, "year", 2012, 2023)
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1e3)
)
# -

funding_ts_uk

percentage_change(funding_ts_uk, 2020, 2021, v=True)
percentage_change(funding_ts_uk, 2021, 2022, v=True)

name = "uk_total_funding_ts"
save_data_table(funding_ts_uk, name, TABLES_DIR)
gs.upload_to_google_sheet(funding_ts_uk, OUTPUTS_TABLE, name, overwrite=True)

# ### UK investment by deal size, over time

# +
# altair stacked bar chart with number of deals, with deal types as color
data = (
    funding_only_early_df_ranges.query("org_id in @companies_df_uk.id.to_list()")
    .groupby(["year", "deal_type"], as_index=True)
    .agg(
        counts=("funding_round_id", "count"),
        total_amount=("raised_amount_gbp", "sum"),
    )
    .reset_index()
    .query("year > 2009")
    .query("year <= 2023")
    # convert to millions
    .assign(total_amount=lambda df: df.total_amount / 1000)
)

# turn the datafrom from long to wide
data_wide_df = (
    data.pivot(index="year", columns="deal_type", values="total_amount")
    .fillna(0)
    .astype(int)
    .reset_index()
)

data_wide_df
# -

name = "uk_total_funding_ranges_ts"
save_data_table(data_wide_df, name, TABLES_DIR)
gs.upload_to_google_sheet(data_wide_df, OUTPUTS_TABLE, name, overwrite=True)

# ### UK investment, by theme

# +
# Total funding per company
companies_funding_period_df = (
    funding_only_early_df.query("year >= 2018 and year <= 2022")
    .groupby("org_id", as_index=False)
    .agg(total_funding_gbp=("raised_amount_gbp", "sum"))
)


uk_companies_funding_theme_period_df = (
    companies_df_uk.merge(
        companies_funding_period_df, left_on="id", right_on="org_id", how="left"
    )
    .merge(childcare_categories_df, left_on="id", right_on="cb_id")
    .drop(["org_id", "cb_id"], axis=1)
)

uk_theme_total_funding_df = (
    uk_companies_funding_theme_period_df.groupby(["theme"], as_index=False)
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .sort_values("total_funding_gbp", ascending=False)
    .reset_index(drop=True)
)

uk_theme_total_funding_df

# +
# uk_companies_funding_theme_period_df.sort_values('total_funding_gbp', ascending=False).head(10)
# -

name = "uk_funding_themes"
save_data_table(uk_theme_total_funding_df, name, TABLES_DIR)
gs.upload_to_google_sheet(
    uk_theme_total_funding_df, OUTPUTS_TABLE, name, overwrite=True
)

# +
sort_order = uk_theme_total_funding_df.theme.to_list()

uk_subtheme_total_funding_df = (
    uk_companies_funding_theme_period_df.groupby(
        ["theme", "subtheme_full"], as_index=False
    )
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .sort_values(["theme", "total_funding_gbp"], ascending=[True, False])
)

uk_subtheme_total_funding_df
# -

name = "uk_funding_subthemes"
save_data_table(uk_subtheme_total_funding_df, name, TABLES_DIR)
gs.upload_to_google_sheet(
    uk_subtheme_total_funding_df, OUTPUTS_TABLE, name, overwrite=True
)

# ### UK investment, by theme and deal size

# +
# Total funding per company
uk_companies_funding_period_df = (
    funding_only_early_df_ranges.query("org_id in @companies_df_uk.id.to_list()")
    .query("year >= 2018 and year <= 2022")
    .groupby(["org_id", "deal_type"])
    .agg(total_funding_gbp=("raised_amount_gbp", "sum"))
    .reset_index()
)

uk_companies_funding_theme_period_df = (
    companies_df_uk.merge(
        uk_companies_funding_period_df, left_on="id", right_on="org_id"
    )
    .merge(childcare_categories_df, left_on="id", right_on="cb_id")
    .drop(["org_id", "cb_id"], axis=1)
)

uk_theme_funding_df = (
    uk_companies_funding_theme_period_df.drop_duplicates(
        subset=["theme", "id", "deal_type"]
    )
    .groupby(["theme", "deal_type"], as_index=False)
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .query("deal_type != 'n/a'")
    .sort_values(["theme", "deal_type"], ascending=[True, True])
)

# turn the datafrom from long to wide
uk_theme_funding_wide_df = (
    uk_theme_funding_df.pivot(
        index="theme", columns="deal_type", values="total_funding_gbp"
    )
    .fillna(0)
    .astype(int)
    .reset_index()
    .merge(uk_theme_total_funding_df, on="theme", how="left")
    .sort_values("total_funding_gbp", ascending=False)
)

uk_theme_funding_wide_df
# -

name = "uk_funding_themes_ranges"
save_data_table(uk_theme_funding_wide_df, name, TABLES_DIR)
gs.upload_to_google_sheet(uk_theme_funding_wide_df, OUTPUTS_TABLE, name, overwrite=True)

# +
uk_subtheme_funding_df = (
    uk_companies_funding_theme_period_df.drop_duplicates(
        subset=["subtheme_full", "id", "deal_type"]
    )
    .groupby(["subtheme_full", "deal_type"], as_index=True)
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .query("deal_type != 'n/a'")
    .reset_index()
)

# #turn the datafrom from long to wide
uk_subtheme_funding_wide_df = (
    uk_subtheme_funding_df.pivot(
        index="subtheme_full", columns="deal_type", values="total_funding_gbp"
    )
    .fillna(0)
    .astype(int)
    .merge(uk_subtheme_total_funding_df, on="subtheme_full", how="left")
    .sort_values("total_funding_gbp", ascending=False)
)

uk_subtheme_funding_wide_df
# -

name = "uk_funding_subthemes_ranges"
save_data_table(uk_subtheme_funding_wide_df, name, TABLES_DIR)
gs.upload_to_google_sheet(
    uk_subtheme_funding_wide_df, OUTPUTS_TABLE, name, overwrite=True
)

# ### Additional details

# Collect all theme time series (NB: Double counting will occur, so don't sum them)
country_ts = []
for country in utils.list_of_select_countries:
    # Get companies in a theme
    companies_country_df = companies_df.query("country == @country").copy()
    country_ids = set(companies_country_df.id.to_list())
    # Get funding for companies in a theme
    funding_country_df = (
        funding_only_early_df.query("org_id in @country_ids").copy()
        # .query("investment_type in @utils.EARLY_STAGE_DEALS")
    )
    # Funding time series
    funding_ts = au.cb_get_all_timeseries(
        companies_country_df, funding_country_df, "year", 2010, 2023
    ).assign(country=country)
    # Collect all time series
    country_ts.append(funding_ts)
# Final dataframe
country_ts = pd.concat(country_ts, ignore_index=True).assign(
    year=lambda df: df.time_period.dt.year
)

top_countries_df = country_funding_wide_df.sort_values(
    "total_funding_gbp", ascending=False
).head(11)
top_countries_df

# +
# variable = 'no_of_rounds'
variable = "raised_amount_gbp_total"

countries = top_countries_df.country.to_list()
initial_values = []
new_values = []
percentage_changes = []
for country in countries:
    ts_df = country_ts.query("country==@country")
    initial_value = ts_df.query("year==2021")[variable].values[0]
    new_value = ts_df.query("year==2022")[variable].values[0]
    percentage_change = au.percentage_change(initial_value, new_value)
    initial_values.append(initial_value)
    new_values.append(new_value)
    percentage_changes.append(percentage_change)

top_countries_change_df = pd.DataFrame(
    {
        "country": countries,
        "initial_value": initial_values,
        "new_value": new_values,
        "percentage_change": percentage_changes,
    }
)

top_countries_change_df.sort_values("percentage_change", ascending=True).reset_index(
    drop=True
)
# -

# # Digital

# Which companies are in digital
digital_df_ = utils.get_digital_companies(companies_df, CB)

combined_categories = {
    "electronics": [
        "hardware",
        "electronics",
        "wearables",
        "internet of things",
        "robotics",
        "smart home",
        "drones",
        "sensor",
    ],
    "social media": [
        "social media",
        "social network",
        "online portals",
        "photo sharing",
        "messaging",
    ],
    "gaming": ["gaming", "video games", "online games", "gamification", "pc games"],
    "mobile": ["mobile", "mobile apps", "apps", "android", "ios", "telecommunications"],
    "audio": [
        "audio",
        "podcast",
        "audiobooks",
        "internet radio",
        "music",
        "musical instruments",
        "music education",
    ],
    "video": ["video streaming", "video", "video on demand", "tv", "broadcasting"],
    "digital entertainment": [
        "digital entertainment",
        "edutainment",
        "media and entertainment",
    ],
    "management software": [
        "management information systems",
        "crm",
        "saas",
        "paas",
        "enterprise software",
        "retail technology",
        "productivity tools",
    ],
    "artificial intelligence": [
        "artificial intelligence",
        "machine learning",
        "speech recognition",
        "natural language processing",
        "predictive analytics",
        "virtual assistant",
    ],
    "data analytics": ["analytics", "big data", "database"],
    "VR/AR": ["virtual reality", "augmented reality", "3d technology"],
    "mobile health": ["mhealth"],
    "ebooks": ["ebooks", "publishing", "reading apps"],
    "fintech": ["mobile payments", "e-commerce platforms"],
    "software development": [
        "web apps",
        "developer platform",
        "developer tools",
        "technical support",
        "application performance management",
    ],
}
categories_to_remove = [
    "software",
    "consumer software",
    "internet",
    "computer",
    "information services",
    "information technology",
    "information and communications technology (ict)",
    "creative agency",
    "art",
    "news",
    "content creators",
]

new_categories_mapping = {
    k: oldk for oldk, oldv in combined_categories.items() for k in oldv
}
new_categories = list(set(list(combined_categories.keys())))


def map_to_new_categories(list_of_categories):
    new_cats = []
    if type(list_of_categories) is list:
        for cat in list_of_categories:
            if cat in new_categories_mapping.keys():
                new_cats.append(new_categories_mapping[cat])
        # dedupe
        new_cats = list(set(new_cats))
    return new_cats


digital_df = digital_df_.assign(
    industry=lambda df: df.industry.apply(map_to_new_categories)
).copy()

industries_orgs = (
    CB.get_company_industries(companies_df)
    .assign(industry=lambda df: df.industry.apply(lambda x: [x]))
    .assign(industry=lambda df: df.industry.apply(map_to_new_categories))
    .assign(n=lambda df: df.industry.apply(len))
    .query("n > 0")
    .drop(columns=["n"])
    .explode("industry")
    .query("industry in @new_categories")
    .merge(companies_df, on=["id", "name"], how="left")
    .drop_duplicates(["id", "name", "industry"])
    .copy()
)

# Get investment time series by company industry categories
# (takes a few minutes to complete)
(
    rounds_by_industry_ts,
    companies_by_industry_ts,
    investment_by_industry_ts,
) = au.investments_by_industry_ts(
    industries_orgs,
    new_categories,
    CB,
    2012,
    2022,
    use_industry_groups=False,
    funding_round_types=utils.EARLY_STAGE_DEALS,
    check_industries=False,
)

# %%
# Number of investment rounds per industry category in the given time period
n_rounds_for_industries = pd.DataFrame(
    rounds_by_industry_ts.reset_index()
    .query("time_period >= 2018 and time_period <= 2022")
    .set_index("time_period")
    .sum(),
    columns=["counts"],
)

# Magnitude vs growth plots
magnitude_growth = au.ts_magnitude_growth(investment_by_industry_ts, 2018, 2022)
# All industries in the digital industry groups
comp_industries = CB.get_company_industries(digital_df)

# +
# Prepare data for the graph
data = (
    # Remove industries without investment or infite growth
    magnitude_growth.query("magnitude!=0")
    .query("growth != inf")
    .sort_values("growth", ascending=False)
    # Add company deal and company counts, and filter by them
    .assign(counts=n_rounds_for_industries["counts"])
    .assign(
        company_counts=comp_industries.groupby("industry").agg(
            company_counts=("id", "count")
        )["company_counts"]
    )
    .query("company_counts>=5 and counts>=10")
    # Drop generic categories
    # .drop(["edtech", "e-learning"])
    .reset_index()
    .rename(columns={"index": "digital_technology"})
    # Create auxillary variables for plotting
    .assign(
        growth=lambda df: df.growth / 100,
        magnitude=lambda df: df.magnitude / 1000,
        Increase=lambda df: df.growth.apply(
            lambda x: "positive" if x >= 0 else "negative"
        ),
    )
)


# %%
fig = (
    alt.Chart(
        data,
        width=300,
        height=300,
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
            scale=alt.Scale(domain=(0, 10)),
        ),
        y=alt.Y(
            "digital_technology:N",
            sort="-x",
            axis=alt.Axis(title="Digital category"),
        ),
        size=alt.Size(
            "magnitude",
            title="Yearly investment (million GBP)",
            legend=alt.Legend(orient="top"),
        ),
        color=alt.Color(
            "Increase",
            sort=["positive", "negative"],
            legend=None,
            scale=alt.Scale(
                domain=["positive", "negative"],
                range=[pu.NESTA_COLOURS[0], pu.NESTA_COLOURS[4]],
            ),
        ),
        tooltip=[
            alt.Tooltip("digital_technology:N", title="Digital technology"),
            alt.Tooltip(
                "magnitude:Q",
                format=",.3f",
                title="Average yearly investment (million GBP)",
            ),
            alt.Tooltip("growth:Q", format=",.0%", title="Growth"),
        ],
    )
)

fig_final = (
    fig.configure_axis(
        gridDash=[1, 7],
        gridColor="grey",
        labelFontSize=pu.FONTSIZE_NORMAL,
        titleFontSize=pu.FONTSIZE_NORMAL,
    )
    .configure_legend(
        labelFontSize=pu.FONTSIZE_NORMAL - 1,
        titleFontSize=pu.FONTSIZE_NORMAL - 1,
    )
    .configure_view(strokeWidth=0)
)

fig_final
# -

# # AI timeseries

ai_orgs = industries_orgs.query("industry == 'artificial intelligence'").id.to_list()

companies_theme_df = companies_df.query("id in @ai_orgs").copy()
# Get funding for companies in a theme
funding_theme_df = funding_only_early_df.query("org_id in @ai_orgs").copy()
# Funding time series
funding_ts = (
    au.cb_get_all_timeseries(companies_theme_df, funding_theme_df, "year", 2010, 2022)
    .assign(theme="artificial intelligence")
    .assign(year=lambda df: df.time_period.dt.year)
    .assign(raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1e3)
)

funding_ts

au.ts_magnitude_growth_(funding_ts, 2018, 2022)

name = "AI_funding_ts"
save_data_table(funding_ts, name, TABLES_DIR)
gs.upload_to_google_sheet(funding_ts, OUTPUTS_TABLE, name, overwrite=True)

companies_theme_df

# ### By deal size

# +
# altair stacked bar chart with number of deals, with deal types as color
data = (
    funding_only_early_df_ranges.query("org_id in @ai_orgs")
    .groupby(["year", "deal_type"], as_index=True)
    .agg(
        counts=("funding_round_id", "count"),
        total_amount=("raised_amount_gbp", "sum"),
    )
    .reset_index()
    .query("year > 2009")
    .query("year <= 2023")
    # convert to millions
    .assign(total_amount=lambda df: df.total_amount / 1000)
)

# turn the datafrom from long to wide
data_wide_df = (
    data.pivot(index="year", columns="deal_type", values="total_amount")
    .fillna(0)
    .astype(int)
    .merge(funding_ts[["year", "raised_amount_gbp_total"]], on="year", how="left")
    .reset_index()
)

data_wide_df
# -

name = "AI_funding_ranges_ts"
save_data_table(data_wide_df, name, TABLES_DIR)
gs.upload_to_google_sheet(data_wide_df, OUTPUTS_TABLE, name, overwrite=True)

companies_df[companies_df.name.str.contains("Lingumi")]

# # Visualisation

import umap
import pandas as pd
from innovation_sweet_spots.utils.embeddings_utils import Vectors
import altair as alt
from innovation_sweet_spots.utils import plotting_utils as pu
from innovation_sweet_spots.getters.preprocessed import (
    get_preprocessed_crunchbase_descriptions,
)

companies_df.id.nunique()

# Load a table with processed company descriptions
processed_texts = get_preprocessed_crunchbase_descriptions()

# +
descriptions_df = (
    processed_texts.query("id in @companies_df.id").drop_duplicates(subset=["id"])
).reset_index()

industries_df = CB.get_company_industries(
    descriptions_df[["id", "name"]], return_lists=True
)

row = descriptions_df.sample().iloc[0]
text = row.description


def fix_industries(text, industries_df, cb_id):
    text_sentences = text.split(".")
    if "Industries" in text_sentences[-1]:
        industries = industries_df[industries_df.id == cb_id].industry.iloc[0]
        try:
            text_sentences[-1] = " Industries: {}".format(", ".join(industries))
        except:
            return text
        return ".".join(text_sentences)
    else:
        return text


for i, row in descriptions_df.iterrows():
    descriptions_df.loc[i, "description"] = fix_industries(
        row.description, industries_df, row.id
    )


# +
# Define constants
EMBEDDINGS_DIR = PROJECT_DIR / "outputs/preprocessed/embeddings"
FILENAME = "childcare_startups_v2"

# Instansiate Vectors class
childcare_vectors = Vectors(
    model_name="all-mpnet-base-v2",
    vector_ids=None,
    filename=FILENAME,
    folder=EMBEDDINGS_DIR,
)
# -

# Make vectors
childcare_vectors.generate_new_vectors(
    new_document_ids=descriptions_df.id.values,
    texts=descriptions_df.description.values,
)

childcare_vectors.save_vectors()

# Download data with partially corrected subthemes
taxonomy_df = gs.download_google_sheet(
    google_sheet_id=utils.AFS_GOOGLE_SHEET_ID_APRIL,
    wks_name="taxonomy_final",
)

colour_map = dict(zip(taxonomy_df.subtheme_full, taxonomy_df.colour))

# +
umap_embeddings = umap.UMAP(
    n_neighbors=50,
    n_components=2,
    metric="euclidean",
    # random_state=21,
    random_state=1000,
).fit_transform(childcare_vectors.vectors)

umap_embeddings_df = pd.DataFrame(umap_embeddings, columns=["umap_1", "umap_2"]).assign(
    id=childcare_vectors.vector_ids
)
# -

umap_embeddings_df.head(3)

companies_df.head(1)

companies_funding_df

companies_df.query("id == '008480fd-6011-4311-99d4-dd1006a66684'")

# +
childcare_embeddings_df = (
    childcare_categories_df.merge(
        companies_df[["id", "name", "cb_url", "country"]],
        right_on="id",
        left_on="cb_id",
        how="left",
    )
    .merge(umap_embeddings_df, on="id", how="left")
    .merge(
        companies_funding_df[["org_id", "total_funding_gbp"]],
        left_on="id",
        right_on="org_id",
        how="left",
    )
    .drop(["subtheme_tag", "keywords", "comments", "id", "org_id"], axis=1)
    .assign(
        is_UK=lambda df: df.country.apply(
            lambda x: "UK" if x == "United Kingdom" else ""
        )
    )
    .assign(
        total_funding_gbp=lambda df: df.total_funding_gbp.apply(
            lambda x: 0.01 if x == 0 else x / 1000
        )
    )
    .fillna(0.01)
)

childcare_embeddings_df

# +
# childcare_embeddings_df[childcare_embeddings_df.name.str.contains('Koru Kids')]
# -

name = "landscape"
save_data_table(childcare_embeddings_df, name, TABLES_DIR)
gs.upload_to_google_sheet(childcare_embeddings_df, OUTPUTS_TABLE, name, overwrite=True)

companies_df[companies_df.name.str.contains("Greenlight")]


# +
# childcare_embeddings_df[childcare_embeddings_df.company_name.str.contains('Noala')]
# -

# # Export table for checking


def investibility_indicator(
    df: pd.DataFrame, funding_threshold_gbp: float = 1
) -> pd.DataFrame:
    """Add an investibility indicator to the dataframe"""
    df["investible"] = funding_threshold_gbp / df["funding_since_2020_gbp"]
    return df


# +
# Get last rounds for each org_id
last_rounds = (
    funding_df.sort_values("announced_on_date")
    .groupby("org_id")
    .last()
    .reset_index()
    .rename(
        columns={
            "org_id": "cb_id",
            "announced_on_date": "last_round_date",
            "raised_amount_gbp": "last_round_gbp",
            "raised_amount_usd": "last_round_usd",
            "post_money_valuation_usd": "last_valuation_usd",
            "investor_count": "last_round_investor_count",
            "cb_url": "deal_url",
        }
    )
    # convert funding to millions
    .assign(last_round_gbp=lambda x: x.last_round_gbp / 1e3)
    .assign(last_round_usd=lambda x: x.last_round_usd / 1e3)
    .assign(last_valuation_usd=lambda x: x.last_valuation_usd / 1e6)
)[
    [
        "cb_id",
        "last_round_date",
        "investment_type",
        "last_round_gbp",
        "last_round_usd",
        "last_valuation_usd",
        "last_round_investor_count",
        "deal_url",
    ]
]


# Get rounds since 2020
last_rounds_since_2020 = (
    funding_df[funding_df.announced_on_date >= "2020-01-01"]
    .groupby("org_id")
    .agg({"raised_amount_gbp": "sum", "funding_round_id": "count"})
    # convert funding to millions
    .assign(raised_amount_gbp=lambda x: x.raised_amount_gbp / 1e3)
    .reset_index()
    .rename(
        columns={
            "org_id": "cb_id",
            "raised_amount_gbp": "funding_since_2020_gbp",
            "funding_round_id": "funding_rounds_since_2020",
        }
    )
)

# Total funding
total_funding = (
    funding_df.groupby("org_id")
    .agg({"raised_amount_gbp": "sum"})
    .reset_index()
    .rename(columns={"org_id": "cb_id", "raised_amount_gbp": "total_funding_gbp"})
    # convert funding to millions
    .assign(total_funding_gbp=lambda x: x.total_funding_gbp / 1e3)
)
# -

# merge all the funding dataframes to company_list
company_list_funding = (
    companies_df.rename(columns={"id": "cb_id"})
    .drop(["total_funding_usd", "last_funding_on"], axis=1)
    .merge(last_rounds, on="cb_id", how="left")
    .merge(last_rounds_since_2020, on="cb_id", how="left")
    .merge(total_funding, on="cb_id", how="left")
    .merge(
        childcare_categories_df[["cb_id", "theme", "subtheme_full"]].rename(
            columns={"subtheme_full": "subtheme"}
        ),
        on="cb_id",
        how="left",
    )
    .merge(
        industries_orgs.groupby("id")
        .agg({"industry": list})
        .reset_index()
        .rename(columns={"industry": "digital_industry"}),
        left_on="cb_id",
        right_on="id",
        how="left",
    )
)
company_list_funding = investibility_indicator(company_list_funding)
# change the order of columns
company_list_funding = company_list_funding[
    [
        "name",
        "theme",
        "subtheme",
        "homepage_url",
        "cb_id",
        "cb_url",
        "country",
        "region",
        "city",
        "rank",
        "short_description",
        "long_description",
        "digital_industry",
        "last_round_date",
        "investment_type",
        "last_round_gbp",
        "last_round_usd",
        "last_valuation_usd",
        "last_round_investor_count",
        "deal_url",
        "funding_since_2020_gbp",
        "funding_rounds_since_2020",
        "total_funding_gbp",
        "investible",
    ]
]
company_list_funding = company_list_funding.fillna(
    {
        "cb_url": "n/a",
    }
)

name = "companies_2023_06_06"
save_data_table(company_list_funding, name, TABLES_DIR)
gs.upload_to_google_sheet(
    company_list_funding,
    "1PmKn1bsM4UOb5EAsBU04RL5S_One6giT1GYAZ5RxZfA",
    name,
    overwrite=True,
)
