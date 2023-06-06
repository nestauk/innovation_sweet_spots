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

# +
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

importlib.reload(utils)


# -

FIGURES_DIR = PROJECT_DIR / "outputs/2023_childcare/figures/"
import innovation_sweet_spots.utils.altair_save_utils as alt_save

AltairSaver = alt_save.AltairSaver(path=FIGURES_DIR)
TABLES_DIR = FIGURES_DIR / "tables"
filetypes = ["html"]


OUTPUTS_TABLE = "107PT9NFeTrIUVhgMKwu-3PlAMhpMg0pH3wPdb0b9pAQ"

# ## Load the data

CB = wu.CrunchbaseWrangler()

childcare_categories_df = pd.read_csv(
    PROJECT_DIR / "outputs/2023_childcare/finals/company_to_subtheme_v2023_05_16.csv"
).drop_duplicates(["cb_id", "subtheme_tag"])

childcare_categories_df


CB.cb_organisations.query("id == '3a29b13a-9c7b-419c-93c2-466de41c3c4b'")

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


# -

# ## Analysis (all companies)

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
funding_df = CB.get_funding_rounds(companies_df)
# -

len(companies_df.id.to_list())

# ### Composition

utils.EARLY_STAGE_DEALS + utils.LATE_STAGE_DEALS

# +
# Total funding per company
companies_funding_df = (
    funding_df.query("investment_type in @utils.EARLY_STAGE_DEALS")
    .groupby("org_id", as_index=False)
    .agg(total_funding_gbp=("raised_amount_gbp", "sum"))
)

companies_funding_theme_df = (
    companies_df.merge(companies_funding_df, left_on="id", right_on="org_id")
    .merge(childcare_categories_df, left_on="id", right_on="cb_id")
    .drop(["org_id", "cb_id"], axis=1)
)
# -

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

theme_counts_df = (
    childcare_categories_df.query("cb_id in @companies_df.id.to_list()")
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


# +
labels_label = "theme"
values_label = "count"
# tooltip = [labels_label, alt.Tooltip(values_label, format=",.3f")]

# %%
fig = (
    alt.Chart(
        theme_counts_df,
        width=200,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(
            f"{values_label}:Q",
            # scale=alt.Scale(domain=[0, 1500]),
            title="Number of companies",
        ),
        alt.Y(
            f"{labels_label}:N",
            sort="-x",
            title="",
        ),
        # tooltip=tooltip,
        color="theme",
    )
)
fig_final = pu.configure_plots(fig)
fig_final

# +
sort_order = theme_counts_df.theme.to_list()

subtheme_counts_df = (
    childcare_categories_df.groupby(["theme", "subtheme_full"])
    .size()
    .reset_index()
    .rename(columns={0: "count"})
    # make theme categorical and provide sort_order list as the sorting order
    .astype({"theme": "category"})
    .assign(theme=lambda x: x.theme.cat.set_categories(sort_order))
    .sort_values(["theme", "count"], ascending=[True, False])
)
subtheme_counts_df
# -

name = "subtheme_counts"
save_data_table(subtheme_counts_df, name, TABLES_DIR)
gs.upload_to_google_sheet(subtheme_counts_df, OUTPUTS_TABLE, name, overwrite=True)

# +
labels_label = "subtheme_full"
values_label = "count"
# tooltip = [labels_label, alt.Tooltip(values_label, format=",.3f")]

# %%
fig = (
    alt.Chart(
        subtheme_counts_df,
        width=200,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(
            f"{values_label}:Q",
            # scale=alt.Scale(domain=[0, 1500]),
            title="Number of companies",
        ),
        alt.Y(
            f"{labels_label}:N",
            sort=subtheme_counts_df[labels_label].to_list(),
            title="",
            # increase label width
            axis=alt.Axis(labelLimit=1000),
        ),
        # tooltip=tooltip,
        color="theme",
    )
)
fig_final = pu.configure_plots(fig)
fig_final
# -

# ### Total investment over time, all companies

# Funding time series
funding_ts = au.cb_get_all_timeseries(
    companies_df, funding_df, "year", 2009, 2023
).assign(year=lambda df: df.time_period.dt.year)

funding_ts

# +
horizontal_label = "Year"
values_label = "Investment (million GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    funding_ts.assign(
        raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
    )
    .query("time_period <= 2023")
    .rename(
        columns={
            "time_period": horizontal_label,
            "raised_amount_gbp_total": values_label,
        }
    )
    .assign(
        **{
            horizontal_label: lambda df: pu.convert_time_period(
                df[horizontal_label], "Y"
            )
        }
    )
)[[horizontal_label, values_label]]

fig = (
    alt.Chart(
        data.query("Year > 2009"),
        width=400,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O", title=""),
        alt.Y(f"{values_label}:Q", scale=alt.Scale(domain=[0, 2500])),
        tooltip=tooltip,
    )
)

fig_final = pu.configure_plots(fig)
fig_final
# -

figure_name = f"parenting_tech_Total_investment"
AltairSaver.save(fig_final, figure_name, filetypes=filetypes)
save_data_table(data, figure_name, TABLES_DIR)

au.percentage_change(
    funding_ts.query("time_period == 2021").raised_amount_gbp_total.iloc[0],
    funding_ts.query("time_period == 2022").raised_amount_gbp_total.iloc[0],
)

au.percentage_change(
    funding_ts.query("time_period == 2018").raised_amount_gbp_total.iloc[0],
    funding_ts.query("time_period == 2022").raised_amount_gbp_total.iloc[0],
)

au.percentage_change(
    funding_ts.query("time_period == 2017").raised_amount_gbp_total.iloc[0],
    funding_ts.query("time_period == 2022").raised_amount_gbp_total.iloc[0],
)

# ### Deal size time series

# +
horizontal_label = "Year"
values_label = "Investment (million GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    funding_ts.assign(
        raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
    )
    .query("time_period <= 2023")
    .rename(
        columns={
            "time_period": horizontal_label,
            "no_of_rounds": values_label,
        }
    )
    .assign(
        **{
            horizontal_label: lambda df: pu.convert_time_period(
                df[horizontal_label], "Y"
            )
        }
    )
)[[horizontal_label, values_label]]

fig = (
    alt.Chart(
        data.query("Year > 2009"),
        width=400,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O", title=""),
        alt.Y(f"{values_label}:Q"),
        tooltip=tooltip,
    )
)

fig_final = pu.configure_plots(fig)
fig_final
# -

funding_df_ = funding_df.assign(
    deal_type=lambda df: df.raised_amount_gbp.apply(utils.deal_amount_to_range_coarse)
).assign(year=lambda df: pd.to_datetime(df.announced_on).dt.year)

# +
# deal_order = [
#     "£0-1M",
#     "£1-4M",
#     "£4-15M",
#     "£15-40M",
#     "£40-100M",
#     "£100-250M",
#     "£250+" "n/a",
# ]
# -

# altair stacked bar chart with number of deals, with deal types as color
data = (
    funding_df_.groupby(["year", "deal_type"], as_index=False)
    .agg(
        counts=("funding_round_id", "count"),
        total_amount=("raised_amount_gbp", "sum"),
    )
    .query("year > 2009")
    .query("year <= 2023")
    # convert to millions
    .assign(total_amount=lambda df: df.total_amount / 1000)
)


data

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

# normalized altair stacked bar chart with number of deals, with deal types as color
fig = (
    alt.Chart(
        data,
        width=400,
        height=300,
    )
    .mark_bar()
    .encode(
        alt.X("year:O", title=""),
        alt.Y(
            "counts:Q",
            title="Number of deals",
            stack="normalize",
            axis=alt.Axis(format="%"),
        ),
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

data

# +
# turn the datafrom from long to wide
data_wide_df = (
    data.pivot(index="year", columns="deal_type", values="total_amount")
    .fillna(0)
    .astype(int)
    .reset_index()
    # .sort_values('Total', ascending=False)
)

data_wide_df
# -

# normalized altair stacked bar chart with number of deals, with deal types as color
fig = (
    alt.Chart(
        data,
        width=400,
        height=300,
    )
    .mark_bar()
    .encode(
        alt.X("year:O", title=""),
        alt.Y(
            "total_amount:Q",
            title="Investment (£ millions)",
            stack="normalize",
            axis=alt.Axis(format="%"),
        ),
        # sort legend
        alt.Color("deal_type:N", title="Deal size", sort=deal_order),
        tooltip=["year", "counts", "deal_type"],
    )
)
pu.configure_plots(fig)

# ## By category

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
    # Get funding for companies in a theme
    funding_theme_df = (
        funding_df.query("org_id in @theme_ids").copy()
        # .query("investment_type in @utils.EARLY_STAGE_DEALS")
    )
    # Funding time series
    funding_ts = au.cb_get_all_timeseries(
        companies_theme_df, funding_theme_df, "year", 2010, 2022
    ).assign(theme=theme)
    # Collect all time series
    themes_ts.append(funding_ts)
# Final dataframe
themes_ts = pd.concat(themes_ts, ignore_index=True).assign(
    year=lambda df: df.time_period.dt.year
)

# Collect all subtheme time series (NB: Double counting will occur, so don't sum them)
subthemes_ts = []
for subtheme in subthemes_list:
    subtheme_ids = subthemes_to_ids[subtheme]
    # Get companies in a theme
    companies_subtheme_df = companies_df.query("id in @subtheme_ids").copy()
    # Get funding for companies in a theme
    funding_subtheme_df = (
        funding_df.query("org_id in @subtheme_ids").copy()
        # .query("investment_type in @utils.EARLY_STAGE_DEALS")
    )
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

cats = themes_list
fig = pu.configure_plots(
    pu.ts_smooth(
        themes_ts.assign(
            raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
        ).query("year < 2023"),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ millions)",
        category_column="theme",
        amount_div=1,
        width=700,
        height=300,
    )
)
fig

figure_name = f"parenting_tech_Total_investment_By_theme"
AltairSaver.save(fig, figure_name, filetypes=filetypes)
save_data_table(themes_ts, figure_name, TABLES_DIR)

# ## By category [early stage]

# change pandas number notation to avoid scientific notation or commas
pd.options.display.float_format = "{:.0f}".format

# ### Total by category

funding_df.columns

# +
# Total funding per company
companies_funding_period_df = (
    funding_df.query("investment_type in @utils.EARLY_STAGE_DEALS")
    .query("year >= 2018 and year <= 2022")
    .groupby("org_id", as_index=False)
    .agg(total_funding_gbp=("raised_amount_gbp", "sum"))
    .assign(deal_type="early_stage")
)


companies_funding_theme_period_df = (
    companies_df.merge(companies_funding_period_df, left_on="id", right_on="org_id")
    .merge(childcare_categories_df, left_on="id", right_on="cb_id")
    .drop(["org_id", "cb_id"], axis=1)
)

# +
theme_funding_df = (
    companies_funding_theme_period_df.groupby(["theme"], as_index=False)
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .sort_values("total_funding_gbp", ascending=False)
)

theme_funding_df

# +
sort_order = theme_funding_df.theme.to_list()

subtheme_funding_df = (
    companies_funding_theme_period_df.groupby(
        ["theme", "subtheme_full"], as_index=False
    )
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .astype({"theme": "category"})
    .assign(theme=lambda x: x.theme.cat.set_categories(sort_order))
    .sort_values(["theme", "total_funding_gbp"], ascending=[True, False])
)

subtheme_funding_df
# -

# ### Total by category, deal type

# +
# Total funding per company
companies_funding_period_total_df = (
    funding_df.query("year >= 2018 and year <= 2022")
    .groupby("org_id", as_index=False)
    .agg(total_funding_gbp=("raised_amount_gbp", "sum"))
    .assign(deal_type="Total")
)

companies_funding_period_early_df = (
    funding_df.query("investment_type in @utils.EARLY_STAGE_DEALS")
    .query("year >= 2018 and year <= 2022")
    .groupby("org_id", as_index=False)
    .agg(total_funding_gbp=("raised_amount_gbp", "sum"))
    .assign(deal_type="Early stage")
)

companies_funding_period_late_df = (
    funding_df.query("investment_type in @utils.LATE_STAGE_DEALS")
    .query("year >= 2018 and year <= 2022")
    .groupby("org_id", as_index=False)
    .agg(total_funding_gbp=("raised_amount_gbp", "sum"))
    .assign(deal_type="Late stage")
)

companies_funding_period_df = pd.concat(
    [
        companies_funding_period_total_df,
        companies_funding_period_early_df,
        companies_funding_period_late_df,
    ]
)

companies_funding_theme_period_df = (
    companies_df.merge(companies_funding_period_df, left_on="id", right_on="org_id")
    .merge(childcare_categories_df, left_on="id", right_on="cb_id")
    .drop(["org_id", "cb_id"], axis=1)
)

# +
theme_funding_df = (
    companies_funding_theme_period_df.groupby(["theme", "deal_type"], as_index=False)
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .sort_values(["deal_type", "total_funding_gbp"], ascending=False)
)

# turn the datafrom from long to wide
theme_funding_wide_df = (
    theme_funding_df.pivot(
        index="theme", columns="deal_type", values="total_funding_gbp"
    )
    .fillna(0)
    .astype(int)
    .reset_index()
    .sort_values("Total", ascending=False)
)

theme_funding_wide_df
# -

# ### Total by categoy, deal size type

import importlib

importlib.reload(utils)

# +
# Total funding per company
companies_funding_period_df = (
    funding_df.query(
        "investment_type in @utils.EARLY_STAGE_DEALS or investment_type in @utils.LATE_STAGE_DEALS "
    )
    .query("year >= 2018 and year <= 2022")
    .assign(
        deal_type=lambda df: df.raised_amount_gbp.apply(
            utils.deal_amount_to_range_coarse
        )
    )
    .groupby(["org_id", "deal_type"], as_index=False)
    .agg(total_funding_gbp=("raised_amount_gbp", "sum"))
)

companies_funding_theme_period_df = (
    companies_df.merge(companies_funding_period_df, left_on="id", right_on="org_id")
    .merge(childcare_categories_df, left_on="id", right_on="cb_id")
    .drop(["org_id", "cb_id"], axis=1)
)

# +
deal_order = ["£0-5M", "£5-20M", "£20-100M", "£100M+"]

theme_funding_df = (
    companies_funding_theme_period_df.drop_duplicates(
        subset=["theme", "id", "deal_type"]
    )
    .groupby(["theme", "deal_type"], as_index=False)
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .sort_values(["deal_type", "total_funding_gbp"], ascending=False)
    .query("deal_type != 'n/a'")
    .astype({"deal_type": "category"})
    .assign(deal_type=lambda x: x.deal_type.cat.set_categories(deal_order))
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
    # .sort_values('Total', ascending=False)
)

theme_funding_wide_df

# +
subtheme_funding_df = (
    companies_funding_theme_period_df.drop_duplicates(
        subset=["subtheme_full", "id", "deal_type"]
    )
    .groupby(["theme", "subtheme_full", "deal_type"], as_index=False)
    .agg(total_funding_gbp=("total_funding_gbp", "sum"))
    .sort_values(["deal_type", "total_funding_gbp"], ascending=False)
    .query("deal_type != 'n/a'")
    .astype({"deal_type": "category"})
    .assign(deal_type=lambda x: x.deal_type.cat.set_categories(deal_order))
    .sort_values(["theme", "subtheme_full", "deal_type"])
)

# turn the datafrom from long to wide
subtheme_funding_wide_df = (
    subtheme_funding_df.pivot(
        index="subtheme_full", columns="deal_type", values="total_funding_gbp"
    )
    .fillna(0)
    .astype(int)
    .reset_index()
    # .sort_values('Total', ascending=False)
)

subtheme_funding_wide_df
# -

# ## Time series

# Funding time series
funding_ts = au.cb_get_all_timeseries(
    companies_df.query("country in @utils.list_of_select_countries"),
    funding_df.query(
        "investment_type in @utils.EARLY_STAGE_DEALS or investment_type in @utils.LATE_STAGE_DEALS"
    ),
    "year",
    2009,
    2023,
).assign(year=lambda df: df.time_period.dt.year)

# +
horizontal_label = "Year"
values_label = "Investment (million GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    funding_ts.assign(
        raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
    )
    .query("time_period < 2023")
    .rename(
        columns={
            "time_period": horizontal_label,
            "raised_amount_gbp_total": values_label,
        }
    )
    .assign(
        **{
            horizontal_label: lambda df: pu.convert_time_period(
                df[horizontal_label], "Y"
            )
        }
    )
)[[horizontal_label, values_label]]

fig = (
    alt.Chart(
        data.query("Year > 2011"),
        width=400,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O", title=""),
        alt.Y(f"{values_label}:Q", scale=alt.Scale(domain=[0, 2500])),
        tooltip=tooltip,
    )
)

fig_final = pu.configure_plots(fig)
fig_final
# -

figure_name = f"parenting_tech_Total_investment_Early_stage"
AltairSaver.save(fig_final, figure_name, filetypes=filetypes)
save_data_table(data, figure_name, TABLES_DIR)

au.percentage_change(
    funding_ts.query("time_period == 2021").raised_amount_gbp_total.iloc[0],
    funding_ts.query("time_period == 2022").raised_amount_gbp_total.iloc[0],
)

au.percentage_change(
    funding_ts.query("time_period == 2020").raised_amount_gbp_total.iloc[0],
    funding_ts.query("time_period == 2022").raised_amount_gbp_total.iloc[0],
)

# Collect all theme time series (NB: Double counting will occur, so don't sum them)
themes_ts = []
for theme in themes_list:
    theme_ids = themes_to_ids[theme]
    # Get companies in a theme
    companies_theme_df = companies_df.query("id in @theme_ids").copy()
    # Get funding for companies in a theme
    funding_theme_df = (
        funding_df.query("org_id in @theme_ids")
        .copy()
        .query("investment_type in @utils.EARLY_STAGE_DEALS")
    )
    # Funding time series
    funding_ts = au.cb_get_all_timeseries(
        companies_theme_df, funding_theme_df, "year", 2010, 2023
    ).assign(theme=theme)
    # Collect all time series
    themes_ts.append(funding_ts)
# Final dataframe
themes_ts = pd.concat(themes_ts, ignore_index=True).assign(
    year=lambda df: df.time_period.dt.year
)

# Collect all subtheme time series (NB: Double counting will occur, so don't sum them)
subthemes_ts = []
for subtheme in subthemes_list:
    subtheme_ids = subthemes_to_ids[subtheme]
    # Get companies in a theme
    companies_subtheme_df = companies_df.query("id in @subtheme_ids").copy()
    # Get funding for companies in a theme
    funding_subtheme_df = (
        funding_df.query("org_id in @subtheme_ids")
        .copy()
        .query("investment_type in @utils.EARLY_STAGE_DEALS")
    )
    # Funding time series
    funding_ts = au.cb_get_all_timeseries(
        companies_subtheme_df, funding_subtheme_df, "year", 2010, 2023
    ).assign(subtheme=subtheme)
    # Collect all time series
    subthemes_ts.append(funding_ts)
# Final dataframe
subthemes_ts = pd.concat(subthemes_ts, ignore_index=True).assign(
    year=lambda df: df.time_period.dt.year
)

cats = themes_list
fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        themes_ts.assign(
            raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
        ).query("year < 2023 and year >= 2012"),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ millions)",
        category_column="theme",
        amount_div=1,
        max_complete_year=2022,
        width=700,
        height=300,
    )
)
fig

# +
# turn the datafrom from long to wide
themes_ts_wide = (
    themes_ts.pivot(index="year", columns="theme", values="raised_amount_gbp_total")
    .fillna(0)
    .astype(int)
    .reset_index()
    # .sort_values('Total', ascending=False)
)

themes_ts_wide
# -

figure_name = f"parenting_tech_Total_investment_By_theme_Early_stage"
AltairSaver.save(fig, figure_name, filetypes=filetypes)
save_data_table(themes_ts, figure_name, TABLES_DIR)

df

au.ts_magnitude_growth_(df, 2018, 2022).assign(theme=theme)

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

dfs.query("variable == 'raised_amount_gbp_total'").sort_values(
    "growth", ascending=False
)

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

dfs.query("variable == 'raised_amount_gbp_total'").sort_values(
    "growth", ascending=False
)

cats = themes_to_subthemes["Content"]
fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        subthemes_ts.assign(
            raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
        ).query("year < 2023 and year >= 2012"),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ millions)",
        category_column="subtheme",
        amount_div=1,
        max_complete_year=2022,
        width=700,
        height=300,
    )
)
fig

cats = themes_to_subthemes["Family support"]
fig = pu.configure_plots(
    pu.ts_smooth_incomplete(
        subthemes_ts.assign(
            raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
        ).query("year < 2023 and year >= 2012"),
        cats,
        variable="raised_amount_gbp_total",
        variable_title="Investment (£ millions)",
        category_column="subtheme",
        amount_div=1,
        max_complete_year=2022,
        width=700,
        height=300,
    )
)
fig

# ## Country comparison

deal_order = ["£0-5M", "£5-20M", "£20-100M", "£100M+"]

# +
# Total funding per company
companies_funding_period_df = (
    funding_df.query(
        "investment_type in @utils.EARLY_STAGE_DEALS or investment_type in @utils.LATE_STAGE_DEALS "
    )
    .query("year >= 2018 and year <= 2022")
    .assign(
        deal_type=lambda df: df.raised_amount_gbp.apply(
            utils.deal_amount_to_range_coarse
        )
    )
    .groupby(["org_id", "deal_type"], as_index=False)
    .agg(
        total_funding_gbp=("raised_amount_gbp", "sum"),
        deal_count=("funding_round_id", "count"),
    )
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
    .astype({"deal_type": "category"})
    .assign(deal_type=lambda x: x.deal_type.cat.set_categories(deal_order))
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
    # .sort_values('Total', ascending=False)
)

country_funding_wide_df

# +
# # altair stacked bar chart with number of deals, with deal types as color
# data = (
#     funding_df_
#     .merge(companies_df[["id", "country"]], left_on="org_id", right_on="id")
#     .query("investment_type in @utils.EARLY_STAGE_DEALS or investment_type in @utils.LATE_STAGE_DEALS ")
#     # .query("investment_type in @utils.EARLY_STAGE_DEALS")
#     .query("year >= 2018 and year <= 2022")
#     .groupby(["country"], as_index=False)
#     .agg(
#         counts=("funding_round_id", "count"),
#         total_amount=("raised_amount_gbp", "sum"),
#     )
#     # convert to millions
#     .assign(total_amount=lambda df: df.total_amount / 1000)
#     .sort_values("total_amount", ascending=False)
#     .reset_index(drop=True)
# )


# +
# companies_df[["id", "country"]].country.unique()
# -

# # UK

# +
funding_df_uk = (
    funding_df.merge(companies_df[["id", "country"]], left_on="org_id", right_on="id")
    .query(
        "investment_type in @utils.EARLY_STAGE_DEALS or investment_type in @utils.LATE_STAGE_DEALS"
    )
    .query("year >= 2012 and year <= 2022")
    .query("country == 'United Kingdom'")
    .copy()
)
companies_df_uk = companies_df.query("country == 'United Kingdom'").copy()

# Funding time series
funding_ts_uk = au.cb_get_all_timeseries(
    companies_df_uk, funding_df_uk, "year", 2012, 2022
).assign(year=lambda df: df.time_period.dt.year)
# -

funding_ts_uk

funding_ts_uk[["raised_amount_gbp_total"]]

# +
horizontal_label = "Year"
values_label = "Investment (million GBP)"
tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

data = (
    funding_ts_uk.assign(
        raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
    )
    .query("time_period < 2023")
    .rename(
        columns={
            "time_period": horizontal_label,
            "raised_amount_gbp_total": values_label,
        }
    )
    .assign(
        **{
            horizontal_label: lambda df: pu.convert_time_period(
                df[horizontal_label], "Y"
            )
        }
    )
)[[horizontal_label, values_label]]

fig = (
    alt.Chart(
        data.query("Year > 2009"),
        width=400,
        height=300,
    )
    .mark_bar(color=pu.NESTA_COLOURS[0])
    .encode(
        alt.X(f"{horizontal_label}:O", title=""),
        alt.Y(f"{values_label}:Q"),
        tooltip=tooltip,
    )
)

fig_final = pu.configure_plots(fig)
fig_final

# +
# horizontal_label = "Year"
# values_label = "Number of deals"
# tooltip = [horizontal_label, alt.Tooltip(values_label, format=",.3f")]

# data = (
#     funding_ts_uk.assign(
#         raised_amount_gbp_total=lambda df: df.raised_amount_gbp_total / 1000
#     )
#     .query("time_period < 2023")
#     .rename(
#         columns={
#             "time_period": horizontal_label,
#             "no_of_rounds": values_label,
#         }
#     )
#     .assign(
#         **{
#             horizontal_label: lambda df: pu.convert_time_period(
#                 df[horizontal_label], "Y"
#             )
#         }
#     )
# )[[horizontal_label, values_label]]

# fig = (
#     alt.Chart(
#         data.query("Year > 2009"),
#         width=400,
#         height=300,
#     )
#     .mark_bar(color=pu.NESTA_COLOURS[0])
#     .encode(
#         alt.X(f"{horizontal_label}:O", title=""),
#         alt.Y(f"{values_label}:Q"),
#         tooltip=tooltip,
#     )
# )

# fig_final = pu.configure_plots(fig)
# fig_final
# -

# Collect all theme time series (NB: Double counting will occur, so don't sum them)
country_ts = []
for country in utils.list_of_select_countries:
    # Get companies in a theme
    companies_country_df = companies_df.query("country == @country").copy()
    country_ids = set(companies_country_df.id.to_list())
    # Get funding for companies in a theme
    funding_country_df = (
        funding_df.query("org_id in @country_ids").copy()
        # .query("investment_type in @utils.EARLY_STAGE_DEALS")
        .query(
            "investment_type in @utils.EARLY_STAGE_DEALS or investment_type in @utils.LATE_STAGE_DEALS "
        )
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

# +
len(companies_df)

importlib.reload(utils)
importlib.reload(au)
# -

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

# +
cats = ["mhealth", "machine learning", "artificial intelligence"]


def map_to_new_categories(list_of_categories):
    new_cats = []
    if type(list_of_categories) is list:
        for cat in list_of_categories:
            if cat in new_categories_mapping.keys():
                new_cats.append(new_categories_mapping[cat])
        # dedupe
        new_cats = list(set(new_cats))
    return new_cats


# -

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

# +
# industries_orgs

# +
# industries_orgs.query("industry=='artificial intelligence'")
# -

importlib.reload(au)

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
    funding_round_types=utils.EARLY_STAGE_DEALS + utils.LATE_STAGE_DEALS,
    check_industries=False,
    # funding_round_types=utils.EARLY_STAGE_DEALS,
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
funding_theme_df = (
    funding_df.query("org_id in @ai_orgs")
    # .query("investment_type in @utils.EARLY_STAGE_DEALS")
    .query(
        "investment_type in @utils.EARLY_STAGE_DEALS or investment_type in @utils.LATE_STAGE_DEALS "
    ).copy()
)
# Funding time series
funding_ts = (
    au.cb_get_all_timeseries(companies_theme_df, funding_theme_df, "year", 2010, 2022)
    .assign(theme="artificial intelligence")
    .assign(year=lambda df: df.time_period.dt.year)
)

len(set(companies_df.id.to_list()))

funding_ts

childcare_categories_df.theme.unique()

theme = "Operations"
theme_ids = childcare_categories_df.query("theme == @theme").cb_id.to_list()
companies_in_theme_df = companies_df.query("id in @theme_ids")

# +
# Which companies are in digital
digital_df = utils.get_digital_companies(companies_in_theme_df, CB)

# Get investment time series by company industry categories
# (takes a few minutes to complete)
(
    rounds_by_industry_ts,
    companies_by_industry_ts,
    investment_by_industry_ts,
) = au.investments_by_industry_ts(
    digital_df.drop("industry", axis=1),
    utils.DIGITAL_INDUSTRIES,
    CB,
    2012,
    2022,
    use_industry_groups=False,
    # funding_round_types=utils.EARLY_STAGE_DEALS + utils.LATE_STAGE_DEALS,
    funding_round_types=utils.EARLY_STAGE_DEALS,
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
    .drop(["edtech", "e-learning"])
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
        height=450,
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
            scale=alt.Scale(domain=(-1, 37)),
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
